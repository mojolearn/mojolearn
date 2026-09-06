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

THE CASE TABLE MIRRORS `mamba/corpus/gen_corpus.py`'s LANDED mamba2 family
(commit 35e7ceaa, `M2_SEED_BASE`/`M2_TENSOR_IDS`/`M2_CASES`) bit for bit --
seed base 0x4D6D6232436F7270 ("Mmb2Corp"), tensor ids 21-31 disjoint from
the Mamba-1 table, cases landing in `mamba/corpus/mamba2/<case>/`. This
file REIMPLEMENTS that table (the Mamba-1 fixture's rule: the check that
loads a corpus case FIRST asserts file bytes == this generator's bytes,
so the two lanes agreeing on the fixture is itself a gate) and the byte
gate is the arbiter if either side drifts. Until the corpus files are
GENERATED (generation is RUN OWED on the corpus side) the corpus arm
REFUSES BY NAME (missing-fixture refusal, not a skip).

One measured-by-design lesson carried from the corpus table: the
STATEPASS witness cases (L = 513, L = 770, the init-states case) override
`A_log` to [-4, -1] so `exp(dA_cs_last)` stays a NORMAL FP32 number
across chunk hops -- under the default range some heads' chunk decay
underflows to +0.0 and the STATEPASS sabotage arms would be bitwise inert
on those cells (a fixture that cannot witness is a clause not gated).

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

comptime M2_TID_X = 21  # x.f32                        [B, L, d_model]
comptime M2_TID_BLOCK_NORM_W = 22  # block_norm.weight.f32 [d_model]
comptime M2_TID_W_IN = 23  # in_proj.weight.f32        [d_in_proj, d_model]
comptime M2_TID_CONV_W = 24  # conv1d.weight.f32       [CD, 1, 4]
comptime M2_TID_CONV_B = 25  # conv1d.bias.f32         [CD]
comptime M2_TID_DT_BIAS = 26  # dt_bias.f32            [H]
comptime M2_TID_A_LOG = 27  # A_log.f32                [H]
comptime M2_TID_D = 28  # D.f32                        [H]
comptime M2_TID_GNORM_W = 29  # norm.weight.f32 (GATED norm) [d_inner]
comptime M2_TID_W_OUT = 30  # out_proj.weight.f32      [d_model, d_inner]
comptime M2_TID_INIT_STATES = 31  # initial_states.f32 [B, H, P, N]

comptime M2_CORPUS_SEED_BASE: UInt64 = 0x4D6D6232436F7270
"""gen_corpus.py::M2_SEED_BASE, ASCII "Mmb2Corp" -- DISTINCT from the
mamba-1 base and with DISJOINT tensor ids, so no (seed, id) pair can
alias; case k's seed is BASE + 0x1000 * k, the mamba-1 rule unchanged."""


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
    """gen_corpus.py::M2_CASES, index for index -- the byte gate depends on
    this table and that one being the SAME table. L = 256 is the exact
    one-chunk boundary, 257 the first crossing, 513 the first three-chunk
    shape, 770 the four-chunk shape; the notes live in gen_corpus.py."""
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
            "m2_base_b3_l4_d64", 3, 4, 64, 2, -1, 0.0, m2_pos_inf(), False
        )
    if k == 3:
        return Mamba2CorpusCase(
            "m2_base_b1_l256_d32", 1, 256, 32, 3, -1, 0.0, m2_pos_inf(), False
        )
    if k == 4:
        return Mamba2CorpusCase(
            "m2_base_b1_l257_d64", 1, 257, 64, 4, -1, 0.0, m2_pos_inf(), False
        )
    if k == 5:
        return Mamba2CorpusCase(
            "m2_comp_b2_l257_d32", 2, 257, 32, 5, -1, 0.0, m2_pos_inf(), False
        )
    if k == 6:
        return Mamba2CorpusCase(
            "m2_comp_row0_b1_l257_d32", 1, 257, 32, 5, 0,
            0.0, m2_pos_inf(), False,
        )
    if k == 7:
        return Mamba2CorpusCase(
            "m2_comp_row1_b1_l257_d32", 1, 257, 32, 5, 1,
            0.0, m2_pos_inf(), False,
        )
    if k == 8:
        # THREE chunks, the two-hop STATEPASS witness; A_log override in
        # m2_case_weights keeps the chunk decay a normal FP32 number.
        return Mamba2CorpusCase(
            "m2_statepass_b1_l513_d32", 1, 513, 32, 8, -1,
            0.0, m2_pos_inf(), False,
        )
    if k == 9:
        return Mamba2CorpusCase(
            "m2_statepass_b2_l770_d64", 2, 770, 64, 9, -1,
            0.0, m2_pos_inf(), False,
        )
    if k == 10:
        # dt_bias IN the distinguishing band [8, 14] (never straddling 20:
        # the mamba-1 adv_softplus_guard lesson); A_log tiny keeps the
        # scan alive.
        return Mamba2CorpusCase(
            "m2_adv_softplus_band_b2_l8_d32", 2, 8, 32, 10, -1,
            0.0, m2_pos_inf(), False,
        )
    if k == 11:
        # A within ulps of 0: the cancellation-prone dA_cs_last - dA_cs.
        return Mamba2CorpusCase(
            "m2_adv_a_near_zero_b3_l64_d32", 3, 64, 32, 11, -1,
            0.0, m2_pos_inf(), False,
        )
    if k == 12:
        # The mamba-1 zero rule applied to x (section 6's sign-bit claims).
        return Mamba2CorpusCase(
            "m2_adv_signed_zeros_b2_l8_d32", 2, 8, 32, 12, -1,
            0.0, m2_pos_inf(), False,
        )
    if k == 13:
        # The z ROWS of in_proj.weight in [-2^30, 2^30]: a saturating z
        # through S8/S21.
        return Mamba2CorpusCase(
            "m2_adv_gate_saturation_b1_l8_d64", 1, 8, 64, 13, -1,
            0.0, m2_pos_inf(), False,
        )
    if k == 14:
        # ACTIVE dt_limit (0.001, 0.1); dt_bias widened to [-9, -1] so
        # BOTH limits bind (the CLAMP_BEFORE_SOFTPLUS witness).
        return Mamba2CorpusCase(
            "m2_adv_dt_limit_b2_l8_d32", 2, 8, 32, 14, -1, 0.001, 0.1, False
        )
    if k == 15:
        # NONZERO initial_states at L = 257: the OTHER STATEPASS witness
        # (a zero-init two-chunk case is bitwise inert there).
        return Mamba2CorpusCase(
            "m2_init_states_b1_l257_d32", 1, 257, 32, 15, -1,
            0.0, m2_pos_inf(), True,
        )
    if k == 16:
        # The corpus's decode-continuation case (kind="decode",
        # prefill_len=260 in gen_corpus.py): as a CASE TABLE row it is the
        # L = 261 prefill, because DEVIATION 786 makes that prefill the
        # decode reference; the decode CHAIN itself is this lane's gate
        # (d), not a fixture.
        return Mamba2CorpusCase(
            "m2_decode_b1_l260p1_d32", 1, 261, 32, 16, -1,
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
    w.norm_w = corpus_tensor(seed, M2_TID_BLOCK_NORM_W, dims.d_model, 0.5, 1.5)
    w.w_in = corpus_tensor(seed, M2_TID_W_IN, dip * dims.d_model, -s_in, s_in)
    w.conv_w = corpus_tensor(seed, M2_TID_CONV_W, cd * M2_D_CONV, -0.5, 0.5)
    w.conv_b = corpus_tensor(seed, M2_TID_CONV_B, cd, -0.125, 0.125)
    w.dt_bias = corpus_tensor(seed, M2_TID_DT_BIAS, h, -7.0, -2.0)
    w.a_log = corpus_tensor(seed, M2_TID_A_LOG, h, 0.0, 2.75)
    w.d_skip = corpus_tensor(seed, M2_TID_D, h, 0.5, 1.5)
    w.gnorm_w = corpus_tensor(seed, M2_TID_GNORM_W, di, 0.5, 1.5)
    w.w_out = corpus_tensor(seed, M2_TID_W_OUT, dims.d_model * di, -s_out, s_out)
    # Overrides, gen_corpus.py::M2_CASES's, branch per case.
    if k == 8 or k == 9 or k == 15:
        # the STATEPASS liveness override: A in [-0.368, -0.018] keeps
        # exp(dA_cs_last) a normal FP32 number over >= 2 chunk hops.
        w.a_log = corpus_tensor(seed, M2_TID_A_LOG, h, -4.0, -1.0)
    if k == 10:  # dt_bias IN THE BAND [8, 14]; A_log tiny keeps dt*A small
        w.dt_bias = corpus_tensor(seed, M2_TID_DT_BIAS, h, 8.0, 14.0)
        w.a_log = corpus_tensor(seed, M2_TID_A_LOG, h, -18.0, -12.0)
    if k == 11:  # A within ulps of 0
        w.a_log = corpus_tensor(seed, M2_TID_A_LOG, h, -18.0, -12.0)
    if k == 14:  # active dt_limit: dt_bias widened so BOTH limits bind
        w.dt_bias = corpus_tensor(seed, M2_TID_DT_BIAS, h, -9.0, -1.0)
    if k == 13:  # the z ROWS of in_proj.weight in [-2^30, 2^30]
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
    if k == 12:
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
    """Where the corpus files for this case live (gen_corpus.py's mamba2
    family layout): `mamba/corpus/mamba2/<case>/<tensor>.f32` plus the
    reference dumps and manifest. The check that reads these REFUSES BY
    NAME when the directory does not exist yet -- generation is RUN OWED
    on the corpus side; a missing fixture is a refusal, never a skip."""
    return String("mamba/corpus/mamba2/") + String(m2_corpus_case(k).name)
