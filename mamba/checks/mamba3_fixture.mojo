# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Hashed fixtures for the Mamba-3 (SISO) identity block, profile
`mojolearn.identical.mamba3.siso.fp32.v1`
(`mamba/IDENTICAL_MAMBA3_CONTRACT.md`). NOT A PORT.

The shape of this file is `mamba/checks/mamba2_fixture.mojo`'s, on purpose,
and it IMPORTS the Mamba-1 fixture's generator machinery
(`corpus_splitmix64`, `corpus_tensor`, `fan_in_scale`, the bit plants)
rather than spelling a second copy: the element rule is
`mojolearn.mamba.corpus.hash.v1` unchanged (contract section 8g -- "hashed
inputs under the element rule with new tensor names"), so the only things
new here are the SEED BASE, the tensor names/ids and the case table.

UNLIKE the mamba2 fixture, THIS TABLE LANDS FIRST: `gen_corpus.py` has no
mamba3 family yet, so this file is the case table the corpus lane must
MIRROR bit for bit (seed base 0x4D6D6233436F7270, ASCII "Mmb3Corp";
tensor ids 41-54, disjoint from the mamba-1 ids 1-11 and the mamba-2 ids
21-31). The byte gate in `mamba3_check.mojo` (file bytes == this
generator's bytes) is the arbiter if either side drifts; until the corpus
files are GENERATED (RUN OWED on the corpus side) the corpus arm REFUSES
BY NAME, never skips.

Scale note (why the ranges are what they are): with `dt_bias` in [-7, -2]
and fan-in-scaled `in_proj`, `dt = softplus(dt_raw + bias)` (NO clamp,
contract S6) spans roughly [9e-4, 0.13]; `dd_A` is a fan-in-scaled
projection output near N(0, ~1), so `A = clamp(-heavy_tail(dd_A),
max=-A_floor)` sits mostly in (-3, -0.3) and `ADT = A * dt` in (-0.4, 0]
-- `exp(da_cs)` stays a normal FP32 number across chunk hops at Q = 64
and the chunked recurrence actually recurses. Each adversarial case names
the sabotage arm or seam clause it exists to witness; a fixture that
cannot witness its arm is the recurring defect contract section 8f's
per-arm witness column exists to prevent.

THE PLANTED WITNESSES, measured-by-design lessons carried from the
siblings:

- `m3_adv_a_floor` (A_FLOOR_UNCLAMPED): the S5 clamp binds ONLY when
  `dd_A < 1 - 1e4` (heavy_tail(x) < 1e-4), which no hashed draw reaches,
  so the dd_A ROWS of in_proj.weight are PLANTED: column 0 = -30000.0,
  every other column +0.0. dd_A is then -30000 * norm.out[:, 0], whose
  sign varies per token, so a healthy fraction of (token, head) cells
  clamp AND a healthy fraction do not; the check COUNTS clamped cells on
  the oracle's clean A.out and refuses zero as VACUOUS.
- `m3_adv_angle_crossing` (ANGLE_MOD_PER_CHUNK / ANGLE_MOD_AT_END): the
  angle ROWS of in_proj.weight are widened so `tanh(angle_raw)` saturates
  toward +-1 (angle rate ~ +-pi) and `dt_bias` is pulled to [-2.5, -1.8]
  (dt ~ 0.08-0.15), so the serial angle recurrence random-walks across
  the [0, 2pi) seam INSIDE the first chunk; the check counts mod
  engagements at non-boundary tokens and refuses zero as VACUOUS
  (a fixture that never crosses is vacuous for these arms -- contract
  8f says so by name).
- `m3_adv_trap_saturating`: the trap ROWS of in_proj.weight in
  [-1024, 1024] push sigma(trap) to both saturation ends (corpus row owed
  by name, contract 8g).
- `m3_init_states` (RESUME_KERNEL_ASSOC witness + the tolerance-checked
  `Input_States` continuation row, contract section 5 claim 2): nonzero
  hashed theta/h/k/v state tensors; theta generated inside [0, 6.25]
  (within [0, 2pi) by construction, the S10 mod's own invariant).
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
)


# ===========================================================================
# Profile constants, contract section 3. Every one is a v1 pin: changing any
# of them is a v2, never a tuning knob (CHUNK_SIZE especially -- mamba2
# DEVIATION 783's standing inherited at the NEW value 64, contract 827/783).
# ===========================================================================

comptime M3_D_STATE = 128
"""N, the QK head dim. mamba3.py:47 default."""
comptime M3_EXPAND = 2
"""mamba3.py:48; d_inner = 2 * d_model."""
comptime M3_HEADDIM = 64
"""P, the V head dim. mamba3.py:49; nheads H = d_inner / headdim."""
comptime M3_NGROUPS = 1
"""G = num_bc_heads. mamba3.py:50, :95. B/C shared across heads by COPY;
rotation is PER HEAD, so post-rotation K/Q differ per head."""
comptime M3_CHUNK_SIZE = 64
"""Q. mamba3.py:64 default ("Recommended: 64 for SISO"). PART OF THE
ARITHMETIC -- mamba2 DEVIATION 783's standing inherited; CHUNK_SIZE_32 is
the falsifier at this value."""
comptime M3_NUM_ROPE_ANGLES = 32
"""R. rope_fraction 0.5 (mamba3.py:53, :98-103): split_tensor_size 64,
num_rope_angles 32; interleaved pairs 32..63 are UNROTATED (structural
cos=1/sin=0, DEVIATION 828)."""
comptime M3_RMS_EPS: Float32 = 1e-5
"""Both the block norm and the B/C norms (mamba3.py:126-127 overrides
rms_norm_ref's 1e-6 default). Bits 0x3727C5AC, contract section 3."""
comptime M3_A_FLOOR: Float32 = 1e-4
"""mamba3.py:57; the S5 clamp bound. Binds only when dd_A < -9999."""
comptime M3_PI: Float32 = 3.141592653589793
"""Float32(pi), bits 0x40490FDB -- pinned BY BITS (contract section 2a);
the check asserts the bits at startup so a literal drift is loud."""
comptime M3_TWO_PI: Float32 = 6.283185307179586
"""Float32(2*pi), bits 0x40C90FDB -- same pin, same startup assert."""


@fieldwise_init
struct Mamba3Dims(Copyable, Movable):
    """One block's shape. Everything below `d_model` is DERIVED from the
    profile constants (mamba3.py:92-107): `d_inner = expand * d_model`,
    `nheads = d_inner / headdim` (so d_model must be a multiple of 32),
    `d_in_proj = 2*d_inner + 2*G*N + 3*nheads + num_rope_angles` with
    column order z | x | B | C | dd_dt | dd_A | trap | angle
    (mamba3.py:106-107)."""

    var d_model: Int
    var d_inner: Int
    var nheads: Int

    @staticmethod
    def of(d_model: Int) raises -> Self:
        if d_model % (M3_HEADDIM // M3_EXPAND) != 0:
            raise Error(
                String("Mamba3Dims: d_model must be a multiple of ")
                + String(M3_HEADDIM // M3_EXPAND)
                + " so that nheads = 2*d_model/64 is whole; got "
                + String(d_model)
            )
        var di = M3_EXPAND * d_model
        return Self(d_model, di, di // M3_HEADDIM)

    def d_in_proj(self) -> Int:
        """Columns of in_proj.out (mamba3.py:107)."""
        return (
            2 * self.d_inner
            + 2 * M3_NGROUPS * M3_D_STATE
            + 3 * self.nheads
            + M3_NUM_ROPE_ANGLES
        )

    # Column offsets of the 8-way split (mamba3.py:106-107, :177-186).
    # The split is a COPY, not a seam.
    def col_z(self) -> Int:
        return 0

    def col_x(self) -> Int:
        return self.d_inner

    def col_b(self) -> Int:
        return 2 * self.d_inner

    def col_c(self) -> Int:
        return 2 * self.d_inner + M3_NGROUPS * M3_D_STATE

    def col_dt(self) -> Int:
        return 2 * self.d_inner + 2 * M3_NGROUPS * M3_D_STATE

    def col_a(self) -> Int:
        return self.col_dt() + self.nheads

    def col_trap(self) -> Int:
        return self.col_dt() + 2 * self.nheads

    def col_angle(self) -> Int:
        return self.col_dt() + 3 * self.nheads


struct Mamba3Weights(Copyable, Movable):
    """One block's parameters, host side, upstream shapes, row-major.
    `in_proj`/`out_proj` carry no bias (mamba3.py:108, :157); there is no
    conv. `mimo_rank = 1` is squeezed out of the B/C bias shapes
    (mamba3.py:121-122 give [H, 1, N]; the SISO call squeezes, :256-257).

        norm_w   [d_model]           the BLOCK norm (block.py Block.forward)
        w_in     [d_in_proj, d_model]  in_proj.weight
        dt_bias  [nheads]            dt_bias (mamba3.py:117)
        bnorm_w  [d_state]           B_norm.weight (RMSNorm, eps 1e-5, :126)
        cnorm_w  [d_state]           C_norm.weight (:127)
        b_bias   [nheads, d_state]   B_bias (ones-init, :121)
        c_bias   [nheads, d_state]   C_bias (ones-init, :122)
        d_skip   [nheads]            D (:140)
        w_out    [d_model, d_inner]  out_proj.weight
    """

    var dims: Mamba3Dims
    var norm_w: List[Float32]
    var w_in: List[Float32]
    var dt_bias: List[Float32]
    var bnorm_w: List[Float32]
    var cnorm_w: List[Float32]
    var b_bias: List[Float32]
    var c_bias: List[Float32]
    var d_skip: List[Float32]
    var w_out: List[Float32]

    def __init__(out self, dims: Mamba3Dims):
        self.dims = dims.copy()
        self.norm_w = List[Float32]()
        self.w_in = List[Float32]()
        self.dt_bias = List[Float32]()
        self.bnorm_w = List[Float32]()
        self.cnorm_w = List[Float32]()
        self.b_bias = List[Float32]()
        self.c_bias = List[Float32]()
        self.d_skip = List[Float32]()
        self.w_out = List[Float32]()


# ===========================================================================
# The mamba3 tensor-name/id table, element rule `mojolearn.mamba.corpus.
# hash.v1` unchanged, NEW names and ids (contract section 8g). Corpus file
# names are `<name>.f32` inside `mamba/corpus/mamba3/<case>/`.
# ===========================================================================

comptime M3_TID_X = 41  # x.f32                          [B, L, d_model]
comptime M3_TID_BLOCK_NORM_W = 42  # block_norm.weight.f32   [d_model]
comptime M3_TID_W_IN = 43  # in_proj.weight.f32          [d_in_proj, d_model]
comptime M3_TID_DT_BIAS = 44  # dt_bias.f32              [H]
comptime M3_TID_BNORM_W = 45  # B_norm.weight.f32        [N]
comptime M3_TID_CNORM_W = 46  # C_norm.weight.f32        [N]
comptime M3_TID_B_BIAS = 47  # B_bias.f32                [H, N]
comptime M3_TID_C_BIAS = 48  # C_bias.f32                [H, N]
comptime M3_TID_D = 49  # D.f32                          [H]
comptime M3_TID_W_OUT = 50  # out_proj.weight.f32        [d_model, d_inner]
comptime M3_TID_INIT_THETA = 51  # init_theta.f32        [B, H, R]
comptime M3_TID_INIT_H = 52  # init_h.f32                [B, H, P, N]
comptime M3_TID_INIT_K = 53  # init_k.f32                [B, H, N]
comptime M3_TID_INIT_V = 54  # init_v.f32                [B, H, P]

comptime M3_CORPUS_SEED_BASE: UInt64 = 0x4D6D6233436F7270
"""ASCII "Mmb3Corp" -- DISTINCT from the mamba-1 ("MambCorp"-family) and
mamba-2 ("Mmb2Corp") bases, with DISJOINT tensor ids, so no (seed, id)
pair can alias; case k's seed is BASE + 0x1000 * k, the mamba-1 rule
unchanged."""


def m3_case_seed(k: Int) -> UInt64:
    return M3_CORPUS_SEED_BASE + UInt64(0x1000) * UInt64(k)


# ===========================================================================
# The case table. Contract section 8g's L set {1, 4, 63, 64, 65, 129, 257}
# lives here (63 just under Q = 64, 64 the exact boundary, 65 the first
# crossing, 129 the first three-chunk shape); d_model in {32, 64};
# adversarial cases by name, each naming what it witnesses.
# ===========================================================================

comptime M3_CORPUS_CASE_COUNT = 18


@fieldwise_init
struct Mamba3CorpusCase(Copyable, Movable):
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
    var has_init_states: Bool
    """True: the four `Input_States` pieces are the case's
    M3_TID_INIT_* tensors (contract section 5 claim 2 -- SUPPORTED,
    tolerance-checked against an unbroken prefill, NEVER bit-gated
    against one; device-vs-oracle on the SAME continuation call is still
    bitwise). False: zeros (upstream's None)."""


def m3_corpus_case(k: Int) raises -> Mamba3CorpusCase:
    """The table the corpus lane must mirror index for index (the byte
    gate depends on this table and gen_corpus.py's future mamba3 table
    being the SAME table)."""
    if k == 0:
        return Mamba3CorpusCase("m3_base_b1_l1_d32", 1, 1, 32, 0, -1, False)
    if k == 1:
        return Mamba3CorpusCase("m3_base_b2_l4_d32", 2, 4, 32, 1, -1, False)
    if k == 2:
        return Mamba3CorpusCase("m3_base_b3_l4_d64", 3, 4, 64, 2, -1, False)
    if k == 3:
        # One row short of the chunk: the largest one-chunk shape.
        return Mamba3CorpusCase("m3_base_b1_l63_d32", 1, 63, 32, 3, -1, False)
    if k == 4:
        # The exact one-chunk boundary (Q = 64).
        return Mamba3CorpusCase("m3_base_b1_l64_d32", 1, 64, 32, 4, -1, False)
    if k == 5:
        # The first chunk crossing; CHUNK_SIZE_32's token-only armed
        # compare runs here too (L > 32, the arm's witness).
        return Mamba3CorpusCase("m3_base_b1_l65_d64", 1, 65, 64, 5, -1, False)
    if k == 6:
        return Mamba3CorpusCase("m3_comp_b2_l65_d32", 2, 65, 32, 6, -1, False)
    if k == 7:
        return Mamba3CorpusCase(
            "m3_comp_row0_b1_l65_d32", 1, 65, 32, 6, 0, False
        )
    if k == 8:
        return Mamba3CorpusCase(
            "m3_comp_row1_b1_l65_d32", 1, 65, 32, 6, 1, False
        )
    if k == 9:
        # THREE working chunks (129 = 2*64 + 1): STATE_TERM_SCALE_FIRST's
        # witness (L > Q with nonzero incoming state at chunks 1 and 2).
        return Mamba3CorpusCase(
            "m3_state_b1_l129_d32", 1, 129, 32, 9, -1, False
        )
    if k == 10:
        return Mamba3CorpusCase(
            "m3_state_b2_l257_d64", 2, 257, 64, 10, -1, False
        )
    if k == 11:
        # dt_bias IN the softplus distinguishing band [8, 14], never
        # straddling 20 (the mamba-1 adv_softplus_guard lesson, inherited
        # verbatim by contract section 4's closing note).
        return Mamba3CorpusCase(
            "m3_adv_softplus_band_b2_l8_d32", 2, 8, 32, 11, -1, False
        )
    if k == 12:
        # The A_FLOOR_UNCLAMPED witness: dd_A rows of w_in PLANTED (see
        # the module docstring); the ONLY region where the S5 clamp binds.
        return Mamba3CorpusCase(
            "m3_adv_a_floor_b2_l8_d32", 2, 8, 32, 12, -1, False
        )
    if k == 13:
        # The ANGLE_MOD_* witness: near-saturated angle rates, dt near
        # 0.1, L = 48 >= 32 -- 2pi-seam crossings INSIDE the first chunk,
        # counted by the check.
        return Mamba3CorpusCase(
            "m3_adv_angle_crossing_b1_l48_d32", 1, 48, 32, 13, -1, False
        )
    if k == 14:
        # trap saturating both directions (corpus row owed by name).
        return Mamba3CorpusCase(
            "m3_adv_trap_saturating_b2_l8_d32", 2, 8, 32, 14, -1, False
        )
    if k == 15:
        # Signed-zero plants BY SIGN BIT (mamba-1 zero rule on x).
        return Mamba3CorpusCase(
            "m3_adv_signed_zeros_b2_l8_d32", 2, 8, 32, 15, -1, False
        )
    if k == 16:
        # Nonzero Input_States continuation: RESUME_KERNEL_ASSOC's
        # witness and the section-5-claim-2 tolerance corpus row.
        return Mamba3CorpusCase(
            "m3_init_states_b1_l65_d32", 1, 65, 32, 16, -1, True
        )
    if k == 17:
        # Gate (d)'s chunk-crossing decode fixture (contract 8d: prefill
        # L = 60, decode through token 70; Q = 64). As a CASE TABLE row it
        # is the L = 70 prefill, because DEVIATION 831 makes that prefill
        # the decode reference; the decode CHAIN itself is gate (d), not
        # a fixture.
        return Mamba3CorpusCase(
            "m3_decode_b1_l60p10_d32", 1, 70, 32, 17, -1, False
        )
    raise Error("m3_corpus_case: no case " + String(k))


def m3_case_weights(k: Int) raises -> Mamba3Weights:
    """The case's parameters, overrides applied -- one branch per
    adversarial case, so a range can never silently drift from the case
    that motivated it (the sibling fixtures' rule)."""
    var c = m3_corpus_case(k)
    var dims = Mamba3Dims.of(c.d_model)
    var seed = m3_case_seed(c.seed_index)
    var w = Mamba3Weights(dims)
    var di = dims.d_inner
    var dip = dims.d_in_proj()
    var h = dims.nheads
    var s_in = fan_in_scale(dims.d_model)
    var s_out = fan_in_scale(di)
    w.norm_w = corpus_tensor(seed, M3_TID_BLOCK_NORM_W, dims.d_model, 0.5, 1.5)
    w.w_in = corpus_tensor(seed, M3_TID_W_IN, dip * dims.d_model, -s_in, s_in)
    w.dt_bias = corpus_tensor(seed, M3_TID_DT_BIAS, h, -7.0, -2.0)
    w.bnorm_w = corpus_tensor(seed, M3_TID_BNORM_W, M3_D_STATE, 0.5, 1.5)
    w.cnorm_w = corpus_tensor(seed, M3_TID_CNORM_W, M3_D_STATE, 0.5, 1.5)
    w.b_bias = corpus_tensor(seed, M3_TID_B_BIAS, h * M3_D_STATE, 0.5, 1.5)
    w.c_bias = corpus_tensor(seed, M3_TID_C_BIAS, h * M3_D_STATE, 0.5, 1.5)
    w.d_skip = corpus_tensor(seed, M3_TID_D, h, 0.5, 1.5)
    w.w_out = corpus_tensor(seed, M3_TID_W_OUT, dims.d_model * di, -s_out, s_out)
    # Overrides, branch per case (the future gen_corpus.py mamba3 family
    # must mirror these; the byte gate is the arbiter).
    if k == 11:  # dt_bias IN THE BAND [8, 14]
        w.dt_bias = corpus_tensor(seed, M3_TID_DT_BIAS, h, 8.0, 14.0)
    if k == 12:
        # PLANT the dd_A rows of w_in: column 0 = -30000.0, others +0.0.
        # Rows [col_a, col_a + H) of the [d_in_proj, d_model] row-major
        # weight are the dd_A channels (mamba3.py:106-107 order).
        var a0 = dims.col_a()
        for r in range(h):
            for jc in range(dims.d_model):
                var idx = (a0 + r) * dims.d_model + jc
                if jc == 0:
                    w.w_in[idx] = Float32(-30000.0)
                else:
                    w.w_in[idx] = Float32(0.0)
    if k == 13:
        # Angle rows of w_in widened so tanh saturates; dt near 0.1.
        var g0 = dims.col_angle()
        var key = corpus_splitmix64(seed ^ (UInt64(M3_TID_W_IN) << 32))
        for r in range(M3_NUM_ROPE_ANGLES):
            for jc in range(dims.d_model):
                var idx = (g0 + r) * dims.d_model + jc
                var hh = corpus_splitmix64(key + UInt64(idx))
                var unit = Float64(Int(hh >> 40)) * 0.000000059604644775390625
                w.w_in[idx] = Float32(-4096.0 + 8192.0 * unit)
        w.dt_bias = corpus_tensor(seed, M3_TID_DT_BIAS, h, -2.5, -1.8)
    if k == 14:
        # trap rows of w_in in [-1024, 1024]: sigma saturates both ways.
        var t0 = dims.col_trap()
        var key = corpus_splitmix64(seed ^ (UInt64(M3_TID_W_IN) << 32))
        for r in range(h):
            for jc in range(dims.d_model):
                var idx = (t0 + r) * dims.d_model + jc
                var hh = corpus_splitmix64(key + UInt64(idx))
                var unit = Float64(Int(hh >> 40)) * 0.000000059604644775390625
                w.w_in[idx] = Float32(-1024.0 + 2048.0 * unit)
    return w^


def m3_case_x(k: Int) raises -> List[Float32]:
    """The case's input, zero rule / parent slice applied."""
    var c = m3_corpus_case(k)
    var seed = m3_case_seed(c.seed_index)
    if c.x_row >= 0:
        var parent = m3_corpus_case(c.seed_index)
        var px = corpus_tensor(
            seed, M3_TID_X, parent.b * parent.l * parent.d_model, -2.0, 2.0
        )
        var out = List[Float32]()
        var row_len = parent.l * parent.d_model
        for i in range(row_len):
            out.append(px[c.x_row * row_len + i])
        return out^
    var x = corpus_tensor(seed, M3_TID_X, c.b * c.l * c.d_model, -2.0, 2.0)
    if k == 15:
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


def _m3_zeros(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(0.0)
    return out^


def m3_case_init_theta(k: Int) raises -> List[Float32]:
    """`Input_States` piece 1, theta [B, H, R]: hashed inside [0, 6.25]
    (within [0, 2pi), the S10 mod's own invariant) when the case carries
    states, zeros otherwise."""
    var c = m3_corpus_case(k)
    var dims = Mamba3Dims.of(c.d_model)
    var n = c.b * dims.nheads * M3_NUM_ROPE_ANGLES
    if c.has_init_states:
        return corpus_tensor(
            m3_case_seed(c.seed_index), M3_TID_INIT_THETA, n, 0.0, 6.25
        )
    return _m3_zeros(n)


def m3_case_init_h(k: Int) raises -> List[Float32]:
    var c = m3_corpus_case(k)
    var dims = Mamba3Dims.of(c.d_model)
    var n = c.b * dims.nheads * M3_HEADDIM * M3_D_STATE
    if c.has_init_states:
        return corpus_tensor(
            m3_case_seed(c.seed_index), M3_TID_INIT_H, n, -1.0, 1.0
        )
    return _m3_zeros(n)


def m3_case_init_k(k: Int) raises -> List[Float32]:
    var c = m3_corpus_case(k)
    var dims = Mamba3Dims.of(c.d_model)
    var n = c.b * dims.nheads * M3_D_STATE
    if c.has_init_states:
        return corpus_tensor(
            m3_case_seed(c.seed_index), M3_TID_INIT_K, n, -1.0, 1.0
        )
    return _m3_zeros(n)


def m3_case_init_v(k: Int) raises -> List[Float32]:
    var c = m3_corpus_case(k)
    var dims = Mamba3Dims.of(c.d_model)
    var n = c.b * dims.nheads * M3_HEADDIM
    if c.has_init_states:
        return corpus_tensor(
            m3_case_seed(c.seed_index), M3_TID_INIT_V, n, -1.0, 1.0
        )
    return _m3_zeros(n)


def m3_corpus_dir(k: Int) raises -> String:
    """Where the corpus files for this case will live (the mamba2 family
    layout, new directory): `mamba/corpus/mamba3/<case>/<tensor>.f32` plus
    the reference dumps and manifest. The check that reads these REFUSES
    BY NAME while the directory does not exist -- generation is RUN OWED
    on the corpus side; a missing fixture is a refusal, never a skip."""
    return String("mamba/corpus/mamba3/") + String(m3_corpus_case(k).name)
