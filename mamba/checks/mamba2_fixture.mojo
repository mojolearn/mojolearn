# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Hashed fixtures for the Mamba-2 (SSD) identity block, profile
`mojolearn.identical.mamba2.fp32.v1` (`mamba/IDENTICAL_MAMBA2_CONTRACT.md`).
NOT A PORT.

The shape of this file is `mamba/checks/mamba_fixture.mojo`'s, on purpose,
and it IMPORTS that file's generator machinery (`corpus_splitmix64`,
`corpus_tensor`, `fan_in_scale`, the bit plants) rather than spelling a
second copy: the element rule is `mojolearn.mamba.corpus.hash.v1` unchanged
(contract section 8g -- "hashed inputs under the element rule with new
tensor names"), so the only things new here are the SEED BASE, the tensor
names/ids and the case table.

THE CASE TABLE IS THIS LANE'S PROPOSAL, PENDING THE CORPUS GENERATOR.
`mamba/corpus/gen_corpus.py` is the corpus lane's file and does not carry
mamba2 cases yet. Contract section 8g pins the element rule and the gate
lengths (L in {1, 4, 256, 257, 513, 770}) and names the adversarial cases;
this file fixes the remaining free choices (seed base, tensor ids, exact
shapes, ranges) so the device gates can run without the corpus directory,
exactly as the Mamba-1 fixture does. When the corpus lane lands its
generator, the byte-compare gate (`mamba2_check.mojo`'s corpus arm: file
bytes == this generator's bytes, asserted FIRST) is the arbiter -- if the
two lanes disagree, that gate fails loudly and ONE of the two tables is
wrong ON THE RECORD, never silently. Until the files exist the corpus arm
REFUSES BY NAME (missing-fixture refusal, not a skip).

Scale note (why the ranges are what they are): with `dt_bias` in [-7, -2]
and fan-in-scaled `in_proj`, `dt = clamp(softplus(dt_raw + bias))` spans
roughly [3e-4, 0.3], `A = -exp(A_log)` with `A_log` in [0, 2.75] sits in
[-15.6, -1] (the `A_init_range=(1,16)` band, contract section 2), so
`exp(dt * A)` is in (0, 1) and the chunked recurrence actually recurses.
The adversarial cases push each boundary on purpose, and each one names
the sabotage arm or seam clause it exists to witness -- a fixture that
cannot witness its arm is the recurring defect the contract's section 8f
table exists to prevent.
"""

from mamba.checks.mamba_fixture import (
    BITS_NEG_ZERO,
    BITS_POS_INF,
    BITS_QNAN,
    bits32_hex,
    corpus_splitmix64,
    corpus_tensor,
    f32_from_bits,
    fan_in_scale,
    mode_name,
    plant_every_kth,
)


# ===========================================================================
# Profile constants, contract section 3. Every one is a v1 pin: changing any
# of them is a v2, never a tuning knob (CHUNK_SIZE especially -- DEVIATION
# 783 says the reduction boundaries ARE the summation order).
# ===========================================================================

comptime M2_D_STATE = 128
"""N. mamba2.py:41 default (the shipped module's, not mamba2_simple's 64)."""
comptime M2_D_CONV = 4
"""mamba2.py:42."""
comptime M2_EXPAND = 2
"""mamba2.py:44; d_inner = 2 * d_model."""
comptime M2_HEADDIM = 64
"""P. mamba2.py:45; nheads H = d_inner / headdim."""
comptime M2_NGROUPS = 1
"""G. mamba2.py:47; B and C broadcast across heads by COPY."""
comptime M2_CHUNK_SIZE = 256
"""Q. mamba2.py:59 default. PART OF THE ARITHMETIC -- DEVIATION 783; the
CHUNK_SIZE_128 sabotage arm is its falsifier."""
comptime M2_RMS_EPS: Float32 = 1e-5
"""Both norms (mamba2.py:144 and the block norm's layer_norm_epsilon).
Bits 0x3727C5AC, contract section 3."""


@fieldwise_init
struct Mamba2Dims(Copyable, Movable):
    """One block's shape. Everything below `d_model` is DERIVED from the
    profile constants (mamba2.py:73-90): `d_inner = expand * d_model`,
    `nheads = d_inner / headdim` (so d_model must be a multiple of 32),
    `d_ssm = d_inner` (d_mlp = 0), `conv_dim = d_ssm + 2 * G * N`,
    `d_in_proj = 2 * d_inner + 2 * G * N + nheads`."""

    var d_model: Int
    var d_inner: Int
    var nheads: Int

    @staticmethod
    def of(d_model: Int) raises -> Self:
        if d_model % (M2_HEADDIM // M2_EXPAND) != 0:
            raise Error(
                String("Mamba2Dims: d_model must be a multiple of ")
                + String(M2_HEADDIM // M2_EXPAND)
                + " so that nheads = 2*d_model/64 is whole; got "
                + String(d_model)
            )
        var di = M2_EXPAND * d_model
        return Self(d_model, di, di // M2_HEADDIM)

    def conv_dim(self) -> Int:
        """CD = d_ssm + 2*G*N (the xBC width, mamba2.py:87)."""
        return self.d_inner + 2 * M2_NGROUPS * M2_D_STATE

    def d_in_proj(self) -> Int:
        """Columns of in_proj.out: z | xBC | dt_raw (mamba2.py:84,
        :211-215 order, d_mlp = 0)."""
        return 2 * self.d_inner + 2 * M2_NGROUPS * M2_D_STATE + self.nheads


struct Mamba2Weights(Copyable, Movable):
    """One block's parameters, host side, upstream shapes, row-major.
    `bias=False, conv_bias=True` (mamba2.py:56-57), so only conv1d carries
    a bias; `dt_bias` is a separate parameter (mamba2.py:117), not a
    Linear bias.

        norm_w   [d_model]              the BLOCK norm (HF Mamba2RMSNorm :591)
        w_in     [d_in_proj, d_model]   in_proj.weight
        conv_w   [conv_dim, d_conv]     conv1d.weight (their [CD,1,4], 1 dropped)
        conv_b   [conv_dim]             conv1d.bias
        dt_bias  [nheads]               dt_bias (mamba2.py:117)
        a_log    [nheads]               A_log (mamba2.py:126; PER HEAD, not per state)
        d_skip   [nheads]               D (D_has_hdim=False, mamba2.py:130)
        gnorm_w  [d_inner]              the GATED norm's weight (RMSNormGated)
        w_out    [d_model, d_inner]     out_proj.weight
    """

    var dims: Mamba2Dims
    var norm_w: List[Float32]
    var w_in: List[Float32]
    var conv_w: List[Float32]
    var conv_b: List[Float32]
    var dt_bias: List[Float32]
    var a_log: List[Float32]
    var d_skip: List[Float32]
    var gnorm_w: List[Float32]
    var w_out: List[Float32]

    def __init__(out self, dims: Mamba2Dims):
        self.dims = dims.copy()
        self.norm_w = List[Float32]()
        self.w_in = List[Float32]()
        self.conv_w = List[Float32]()
        self.conv_b = List[Float32]()
        self.dt_bias = List[Float32]()
        self.a_log = List[Float32]()
        self.d_skip = List[Float32]()
        self.gnorm_w = List[Float32]()
        self.w_out = List[Float32]()


# ===========================================================================
# The mamba2 tensor-name/id table, element rule `mojolearn.mamba.corpus.
# hash.v1` (the mamba-1 spec's rule, NEW names -- contract section 8g).
# Corpus file names are `<name>.f32` inside `mamba/corpus/<case>/`.
# ===========================================================================

comptime M2_TID_X = 1  # x.f32                  [B, L, d_model]
comptime M2_TID_W_IN = 2  # in_proj.weight.f32  [d_in_proj, d_model]
comptime M2_TID_CONV_W = 3  # conv1d.weight.f32 [conv_dim, 4]
comptime M2_TID_CONV_B = 4  # conv1d.bias.f32   [conv_dim]
comptime M2_TID_DT_BIAS = 5  # dt_bias.f32      [nheads]
comptime M2_TID_A_LOG = 6  # A_log.f32          [nheads]
comptime M2_TID_D = 7  # D.f32                  [nheads]
comptime M2_TID_GNORM_W = 8  # norm_gated.weight.f32 [d_inner]
comptime M2_TID_NORM_W = 9  # norm.weight.f32   [d_model]
comptime M2_TID_W_OUT = 10  # out_proj.weight.f32 [d_model, d_inner]
comptime M2_TID_INIT_STATES = 12  # initial_states.f32 [B, H, P, N]

comptime M2_CORPUS_SEED_BASE: UInt64 = 0x4D616D6261324353
"""ASCII "Mamba2CS". DISTINCT from the mamba-1 base 0x4D616D6261436F72 so no
mamba2 tensor can collide with a mamba1 tensor's stream; case k's seed is
BASE + 0x1000 * k, the mamba-1 rule unchanged."""


def m2_case_seed(k: Int) -> UInt64:
    return M2_CORPUS_SEED_BASE + UInt64(0x1000) * UInt64(k)


# ===========================================================================
# The case table. Contract section 3's gate shapes: B in {1,2,3},
# L in {1, 4, 256, 257, 513, 770}, d_model in {32, 64}; section 8g's
# adversarial cases by name. Each case says WHAT IT WITNESSES.
# ===========================================================================

comptime M2_CORPUS_CASE_COUNT = 17


@fieldwise_init
struct Mamba2CorpusCase(Copyable, Movable):
    var name: StaticString
    var b: Int
    var l: Int
    var d_model: Int
    var seed_index: Int
    """The index whose seed generates the tensors (differs from the case's
    own index only for the comp_row slices, which use the parent's)."""
    var x_row: Int
    """-1: x is the case's own tensor. >= 0: that batch row of the parent
    case's x (the composition slices)."""
    var dt_lo: Float32
    var dt_hi: Float32
    """The runtime `dt_limit` input. (0.0, +inf) is the profile default and
    the clamp is then PRESENT AND INERT (contract S9); the active case
    carries (0.001, 0.1)."""
    var has_init_states: Bool
    """True: `initial_states` is the case's M2_TID_INIT_STATES tensor
    (ssd_minimal.py:64-66's chunk -1 state). False: zeros."""


def m2_pos_inf() -> Float32:
    return f32_from_bits(BITS_POS_INF)


def m2_corpus_case(k: Int) raises -> Mamba2CorpusCase:
    # Base shapes: every gate length appears at least once; L = 256 is the
    # exact-one-chunk boundary, 257 the first crossing, 513 the first
    # THREE-chunk shape (the two-hop decay STATEPASS_MATRIX needs), 770 the
    # four-chunk composition shape.
    if k == 0:
        return Mamba2CorpusCase(
            "m2_base_b1_l1_d32", 1, 1, 32, 0, -1, 0.0, m2_pos_inf(), False
        )
    if k == 1:
        return Mamba2CorpusCase(
            "m2_base_b2_l4_d32", 2, 4, 32, 1, -1, 0.0, m2_pos_inf(), False
        )
    if k == 2:
        return Mamba2CorpusCase(
            "m2_base_b1_l256_d32", 1, 256, 32, 2, -1, 0.0, m2_pos_inf(), False
        )
    if k == 3:
        return Mamba2CorpusCase(
            "m2_base_b3_l257_d32", 3, 257, 32, 3, -1, 0.0, m2_pos_inf(), False
        )
    if k == 4:
        # THREE chunks: the shape STATEPASS_MATRIX / STATEPASS_UNFUSED need
        # (contract 8f: a zero-init two-chunk case is bitwise inert there).
        return Mamba2CorpusCase(
            "m2_base_b1_l513_d32", 1, 513, 32, 4, -1, 0.0, m2_pos_inf(), False
        )
    if k == 5:
        return Mamba2CorpusCase(
            "m2_base_b2_l770_d32", 2, 770, 32, 5, -1, 0.0, m2_pos_inf(), False
        )
    if k == 6:
        return Mamba2CorpusCase(
            "m2_base_b1_l4_d64", 1, 4, 64, 6, -1, 0.0, m2_pos_inf(), False
        )
    if k == 7:
        return Mamba2CorpusCase(
            "m2_base_b1_l257_d64", 1, 257, 64, 7, -1, 0.0, m2_pos_inf(), False
        )
    if k == 8:
        # dt_bias drawn from [8, 14]: the softplus guard's DISTINGUISHING
        # band (the mamba-1 adv_softplus_guard lesson: a 20-straddling
        # fixture is vacuous; plant IN THE BAND).
        return Mamba2CorpusCase(
            "m2_adv_softplus_band_b2_l8_d32", 2, 8, 32, 8, -1,
            0.0, m2_pos_inf(), False,
        )
    if k == 9:
        # A_log in [-18, -12]: A within ulps of 0, decay near 1 -- the
        # cancellation-prone `dA_cs_last - dA_cs` (S15) and the segsum's
        # small-argument band.
        return Mamba2CorpusCase(
            "m2_adv_a_near_zero_b3_l8_d32", 3, 8, 32, 9, -1,
            0.0, m2_pos_inf(), False,
        )
    if k == 10:
        # The mamba-1 zero rule applied to x: whole +0.0 tokens, whole -0.0
        # tokens, scattered -0.0 -- section 6's sign-bit claims.
        return Mamba2CorpusCase(
            "m2_adv_signed_zeros_b2_l8_d32", 2, 8, 32, 10, -1,
            0.0, m2_pos_inf(), False,
        )
    if k == 11:
        # The gate half of in_proj.weight in [-2^30, 2^30]: a saturating z
        # through S8/S21 (the adv_gate_saturation shape; its corpus
        # tolerance is calibrated per case, never carried over).
        return Mamba2CorpusCase(
            "m2_adv_gate_saturation_b1_l8_d64", 1, 8, 64, 11, -1,
            0.0, m2_pos_inf(), False,
        )
    if k == 12:
        # ACTIVE dt_limit (0.001, 0.1): both limits BIND on real cells
        # (default dt spans ~[3e-4, 0.3]). The CLAMP_BEFORE_SOFTPLUS arm's
        # witnessing fixture (contract 8f) and S9's active-clamp path.
        return Mamba2CorpusCase(
            "m2_adv_dt_limit_b2_l8_d32", 2, 8, 32, 12, -1,
            0.001, 0.1, False,
        )
    if k == 13:
        # NONZERO initial_states at L = 257 (two chunks): the OTHER
        # witnessing fixture the STATEPASS arms need (contract 8f), and
        # ssd_minimal:64-66's chunk -1 semantics exercised.
        return Mamba2CorpusCase(
            "m2_adv_initstate_b1_l257_d32", 1, 257, 32, 13, -1,
            0.0, m2_pos_inf(), True,
        )
    if k == 14:
        return Mamba2CorpusCase(
            "m2_comp_b2_l257_d32", 2, 257, 32, 14, -1, 0.0, m2_pos_inf(), False
        )
    if k == 15:
        return Mamba2CorpusCase(
            "m2_comp_row0_b1_l257_d32", 1, 257, 32, 14, 0,
            0.0, m2_pos_inf(), False,
        )
    if k == 16:
        return Mamba2CorpusCase(
            "m2_comp_row1_b1_l257_d32", 1, 257, 32, 14, 1,
            0.0, m2_pos_inf(), False,
        )
    raise Error("m2_corpus_case: no case " + String(k))


def m2_case_weights(k: Int) raises -> Mamba2Weights:
    """The case's parameters, overrides applied -- one branch per
    adversarial case, so a range can never silently drift from the case
    that motivated it (the mamba-1 fixture's rule)."""
    var c = m2_corpus_case(k)
    var dims = Mamba2Dims.of(c.d_model)
    var seed = m2_case_seed(c.seed_index)
    var w = Mamba2Weights(dims)
    var di = dims.d_inner
    var cd = dims.conv_dim()
    var dip = dims.d_in_proj()
    var h = dims.nheads
    var s_in = fan_in_scale(dims.d_model)
    var s_out = fan_in_scale(di)
    w.norm_w = corpus_tensor(seed, M2_TID_NORM_W, dims.d_model, 0.5, 1.5)
    w.w_in = corpus_tensor(seed, M2_TID_W_IN, dip * dims.d_model, -s_in, s_in)
    w.conv_w = corpus_tensor(seed, M2_TID_CONV_W, cd * M2_D_CONV, -0.5, 0.5)
    w.conv_b = corpus_tensor(seed, M2_TID_CONV_B, cd, -0.125, 0.125)
    w.dt_bias = corpus_tensor(seed, M2_TID_DT_BIAS, h, -7.0, -2.0)
    w.a_log = corpus_tensor(seed, M2_TID_A_LOG, h, 0.0, 2.75)
    w.d_skip = corpus_tensor(seed, M2_TID_D, h, 0.5, 1.5)
    w.gnorm_w = corpus_tensor(seed, M2_TID_GNORM_W, di, 0.5, 1.5)
    w.w_out = corpus_tensor(seed, M2_TID_W_OUT, dims.d_model * di, -s_out, s_out)
    if k == 8:  # dt_bias IN THE BAND [8, 14], not straddling 20
        w.dt_bias = corpus_tensor(seed, M2_TID_DT_BIAS, h, 8.0, 14.0)
    if k == 9:  # A within ulps of 0
        w.a_log = corpus_tensor(seed, M2_TID_A_LOG, h, -18.0, -12.0)
    if k == 11:  # the z half of in_proj.weight in [-2^30, 2^30]
        # Column order is z | xBC | dt (mamba2.py:211-215): the z channels
        # are ROWS [0, d_inner) of w_in ([d_in_proj, d_model] row-major).
        var n = dip * dims.d_model
        var half = di * dims.d_model
        var key = corpus_splitmix64(seed ^ (UInt64(M2_TID_W_IN) << 32))
        var out = List[Float32]()
        for i in range(n):
            var hh = corpus_splitmix64(key + UInt64(i))
            var unit = Float64(Int(hh >> 40)) * 0.000000059604644775390625
            if i < half:
                out.append(Float32(-1073741824.0 + 2147483648.0 * unit))
            else:
                out.append(Float32(-s_in + (2.0 * s_in) * unit))
        w.w_in = out^
    return w^


def m2_case_x(k: Int) raises -> List[Float32]:
    """The case's input, zero rule / parent slice applied."""
    var c = m2_corpus_case(k)
    var seed = m2_case_seed(c.seed_index)
    if c.x_row >= 0:
        var parent = m2_corpus_case(c.seed_index)
        var px = corpus_tensor(
            seed, M2_TID_X, parent.b * parent.l * parent.d_model, -2.0, 2.0
        )
        var out = List[Float32]()
        var row_len = parent.l * parent.d_model
        for i in range(row_len):
            out.append(px[c.x_row * row_len + i])
        return out^
    var x = corpus_tensor(seed, M2_TID_X, c.b * c.l * c.d_model, -2.0, 2.0)
    if k == 10:
        # The mamba-1 zero rule verbatim (gen_corpus.py::apply_zero_rule):
        # whole +0.0 tokens (t%4==0), whole -0.0 tokens (t%4==2), scattered
        # -0.0 at flat i%7==3 elsewhere.
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


def m2_case_init_states(k: Int) raises -> List[Float32]:
    """`initial_states` [B, H, P, N] for the case: the hashed tensor when
    `has_init_states`, zeros otherwise (upstream's `None`,
    ssd_minimal.py:64-66)."""
    var c = m2_corpus_case(k)
    var dims = Mamba2Dims.of(c.d_model)
    var n = c.b * dims.nheads * M2_HEADDIM * M2_D_STATE
    if c.has_init_states:
        return corpus_tensor(
            m2_case_seed(c.seed_index), M2_TID_INIT_STATES, n, -1.0, 1.0
        )
    var out = List[Float32]()
    for _ in range(n):
        out.append(0.0)
    return out^


def m2_corpus_dir(k: Int) raises -> String:
    """Where the corpus lane's files for this case will live, per the
    mamba-1 layout: `mamba/corpus/<case>/<tensor>.f32` plus `ref32/`,
    `ref64/` and `manifest.json`. The check that reads these REFUSES BY
    NAME when the directory does not exist yet -- a missing fixture is a
    refusal, never a skip."""
    return String("mamba/corpus/") + String(m2_corpus_case(k).name)
