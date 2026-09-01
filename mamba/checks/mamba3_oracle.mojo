# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The host Float32 oracle of one Mamba-3 (SISO) block under profile
`mojolearn.identical.mamba3.siso.fp32.v1`, and the Float64 tolerance
reference.

NOT A PORT -- the reference library ships no oracle; the ALGORITHM below
is theirs, cited seam by seam in `mamba/IDENTICAL_MAMBA3_CONTRACT.md`
(normative math `tests/ops/triton/test_mamba3_siso.py::mamba3_siso_fwd_ref`
:149-340; block order `mamba_ssm/modules/mamba3.py::Mamba3.forward`
:160-278; chunked schedule SHAPE `ops/triton/mamba3/mamba3_siso_fwd.py`,
DEVIATION 827; state-spaces/mamba `e9594ce`). This file is the contract's
arithmetic order written on the CPU through the SAME seam functions the
sibling lanes certified (`identical_mul_add`, `ftz`, `identical_exp`,
`identical_div`, `identical_rsqrt`, `identical_silu`, `identical_softplus`,
`identical_sigmoid`, `identical_tanh`, `identical_clamp`,
`portable_cosf`/`portable_sinf` -- DEVIATION 820's pair, whose device
certification is a PRECONDITION, contract phase M3-0, RUN OWED) plus GEMM
v1's `gemm_oracle`, so the device card of
`mamba/impl/mamba_ssm/modules/mamba3.mojo` / `ops/mamba3_siso.mojo` is
diffed against it bitwise. The device kernels are an INDEPENDENT
transcription of the same order; nothing below is imported by them except
the seam functions themselves (one arithmetic, two spellings of the loops
around it -- the sibling lanes' rule, unchanged).

CONTRACT DEVIATIONS IMPLEMENTED (827-831 the contract's, cited never
renumbered): 827 (the chunked two-phase schedule at Q = 64, the kernel's,
not the unchunked reference's); 828 (portable trig pair on both paths,
interleaved (2i, 2i+1) pairing, UNFUSED two-product rotation, STRUCTURAL
identity for pairs >= num_rope_angles); 829 (per-token serial angle
recurrence, mod 2pi applied EVERY step, mod composed from identical_div /
exact floor / pinned 2pi bits / one subtract); 830 (the diagonal rides
the pre-rotation QK dot times gamma; the reference's include-then-subtract
spelling is the DIAG_INCLUDE_SUBTRACT required-RED arm); 831 (decode is
prefill resumption via the buffer construction below; the upstream
four-piece `Input_States` continuation is SUPPORTED and tolerance-checked,
never bit-gated against an unbroken prefill).

DEVIATION 832 (NEW, this file owns it; the impl and check cite it) --
THE SEALED-CHUNK RESUMPTION BOUNDARY, THE WORKING-SEQUENCE STAGE SHAPES,
AND GATE (d)'s COMPARABILITY CLAUSE. Three connected clauses, all forced
by the trapezoid's SHIFTED loads (contract section 3: token t's K-row
scale is `gamma_t + beta'_{t+1}`, which needs token t+1):

  (i) A chunk is SEALED -- its fold into the carried state h is final --
      only once the FIRST TOKEN OF THE NEXT CHUNK exists, because its
      last row's scale reads that token's dt and sigma(trap). The carried
      boundary h is therefore the state ENTERING THE LAST WORKING CHUNK
      (= after all sealed chunks; with C = ceil(t_work/Q) working chunks,
      exactly C-1 are sealed), and the intra-chunk buffer keeps the last
      working chunk's r = t_work - (C-1)*Q rows, r in [1, Q] -- it NEVER
      empties after the first token, unlike mamba2's (whose conv-less
      sibling rule was r = t_work mod Q). A construction that folded a
      chunk the moment it filled would fold its last K-row at
      `pinned_mul(k, gamma)` and owe the beta' leg a second rounding
      later, which is exactly the split the contract proves unequal
      (section 5 claim 2); the sealed rule folds every sealed row ONCE at
      `pinned_mul(k, ftz(gamma + beta'))`, which is what makes gate (d)'s
      decode == prefill a theorem. Buffered rows carry their
      ROTATED-UNSCALED q/k, v, dt, sigma(trap) and ADT (contract section
      5's list); every quantity rebuilt from them is a pure function of
      the same bits.
 (ii) Chunk-shaped stages (`dacs.out`, `seg.L`, and the compare-only
      `pass.states`) cover the WORKING chunks (buffer ++ new rows), which
      on a fresh prefill coincide with contract section 7 verbatim --
      mamba2 DEVIATION 790's clause at the new shapes. Token-shaped SSD
      stages record the NEW tokens only, sliced (a copy).
(iii) Comparability under gate (d): every token-shaped stage EXCEPT
      `trap.scale` and `kscale.out` is PREFIX-STABLE (token t's value in
      a prefill of any L > t equals its value in a prefill of t+1 -- the
      shifted operand it reads is at most token t) and compares bitwise
      per token. `trap.scale` and `kscale.out` are NOT prefix-stable: a
      token's beta' leg is +0.0 while it is the last token and becomes
      real when its successor arrives, so those two compare ONLY at the
      final decoded token (where both sides' shifted operand is the
      structural +0.0 past the sequence end). Chunk-shaped stages compare
      at the final token against the prefill's chunks (L-2)//Q onward
      (same fill by construction); the carried state and the four reports
      compare at the final token. Anyone who "fixes" the gate to compare
      trap.scale per token has misread the trapezoid. No arithmetic moves
      under this deviation; it is a construction-and-addressing clause.

DEVIATION 833 (NEW, this file owns it) -- THE Y-COMBINE ASSOCIATION.
The contract's S18 takes `Y` (state term + intra-chunk attention) as
given without pinning the one add that forms it. The shipped kernel's
accumulator order is state-term-first (`mamba3_siso_fwd.py`:406-418:
`acc_o = dot(q, states) * exp(da_cs)` THEN `acc_o += dot(s, v)`), so the
profile pins `y = ftz(ystate + yintra)` -- ONE rounding, ystate the left
operand -- before S18's `ftz(y + pinned_mul(ftz(D + qk_gamma), v))`.
Recorded as a deviation because the contract is silent and the reference
(unchunked) has no corresponding add at all.

DECODE IS PREFILL RESUMPTION (DEVIATION 831 + 832). ONE function serves
both paths: `mamba3_block_oracle` takes the carried state (theta, sealed
boundary h, the last working chunk's buffered rows), rebuilds the working
sequence, and runs the chunked schedule over it. A prefill is the same
call with a fresh zero state. The upstream per-token step recurrence
(`mamba3_siso_step_ref`:119-127) is NOT here; it lives in the impl file
as the required-RED arm STEP_UPSTREAM_RECURRENCE.

The Float64 reference at the bottom is the per-token trapezoidal
recurrence (`mamba3_siso_step_ref`'s alpha/beta/gamma algebra, which the
chunked schedule factorizes exactly) in double precision. It is a
TOLERANCE instrument, never a bitwise one.
"""

from checks.numerics import (
    ftz,
    identical_clamp,
    identical_div,
    identical_exp,
    identical_mul_add,
    identical_rsqrt,
    identical_sigmoid,
    identical_silu,
    identical_softplus,
    identical_tanh,
    portable_cosf,
    portable_sinf,
)
from gemm.checks.gemm_oracle import OP_NN, OP_NT, OP_TN, gemm_oracle
from mamba.checks.mamba_oracle import refuse_nonfinite
from mamba.checks.mamba3_fixture import (
    BITS_POS_INF,
    M3_A_FLOOR,
    M3_CHUNK_SIZE,
    M3_D_STATE,
    M3_HEADDIM,
    M3_NUM_ROPE_ANGLES,
    M3_PI,
    M3_RMS_EPS,
    M3_TWO_PI,
    Mamba3Dims,
    Mamba3Weights,
    f32_from_bits,
)


def pinned_mul(a: Float32, b: Float32) -> Float32:
    """DEVIATION 720's construction, the sibling oracles' copy: a MULTIPLY
    no codegen may contract into a neighboring add, spelled
    `identical_mul_add(a, b, -0.0)`."""
    return identical_mul_add(a, b, Float32(-0.0))


def m3_neg_inf() -> Float32:
    """-inf for S5's `identical_clamp(., -inf, -A_floor)` lower bound
    (which never binds; the clamp primitive is used whole, DEVIATION
    788's requirement inherited)."""
    return -f32_from_bits(BITS_POS_INF)


def m3_mod_2pi(x: Float32) -> Float32:
    """DEVIATION 829's composed mod: `ftz(x - pinned_mul(2pi,
    floor(identical_div(x, 2pi))))`, floor exact, 2pi the pinned bits.
    Not a primitive -- each side spells its own copy of this composition
    (the pinned_mul rule)."""
    from std.math import floor

    return ftz(
        x - pinned_mul(M3_TWO_PI, floor(identical_div(x, M3_TWO_PI)))
    )


def m3_heavy_tail_a(dd_a: Float32) -> Float32:
    """Seam S5: `A = identical_clamp(-heavy_tail(dd_A), -inf, -A_floor)`,
    the piecewise spelling (x >= 0 -> ftz(1+x); x < 0 ->
    identical_div(1, ftz(1-x)); negate exact; clamp). The reference's
    branchless `clamp_min + reciprocal(1 - clamp_max)` sum is bit-equal
    (the inactive term is an exact +0.0 or the exact x+1 commutation) --
    contract S5, recorded 782-style."""
    var x = ftz(dd_a)
    var ht: Float32
    if x >= Float32(0.0):
        ht = ftz(Float32(1.0) + x)
    else:
        ht = ftz(identical_div(Float32(1.0), ftz(Float32(1.0) - x)))
    return ftz(identical_clamp(-ht, m3_neg_inf(), -M3_A_FLOOR))


def m3_refuse_bad_inputs(
    w: Mamba3Weights, x: List[Float32], state: Mamba3State
) raises:
    """Contract section 6 (mamba1 section 6 verbatim): refusal BY NAME and
    BY BITS before any recorded stage. Structural zeros (S16's triangle,
    S13's unrotated pairs, the padded rows, the last token's shifted
    +0.0) never exist as computed nonfinites, so nothing here needs an
    exemption. Non-Float32 surfaces (the shipped bf16 casts), is_mimo,
    is_outproj_norm, rope_fraction != 0.5, ngroups > 1 and varlen are
    refused STRUCTURALLY: this API carries no such knob and no such
    dtype, which is the profile's refusal-by-name (contract section 3)."""
    refuse_nonfinite("x", x)
    refuse_nonfinite("norm.weight", w.norm_w)
    refuse_nonfinite("in_proj.weight", w.w_in)
    refuse_nonfinite("dt_bias", w.dt_bias)
    refuse_nonfinite("B_norm.weight", w.bnorm_w)
    refuse_nonfinite("C_norm.weight", w.cnorm_w)
    refuse_nonfinite("B_bias", w.b_bias)
    refuse_nonfinite("C_bias", w.c_bias)
    refuse_nonfinite("D", w.d_skip)
    refuse_nonfinite("out_proj.weight", w.w_out)
    refuse_nonfinite("state.theta", state.theta)
    refuse_nonfinite("state.h", state.h)
    refuse_nonfinite("state.buf_qrot", state.buf_qrot)
    refuse_nonfinite("state.buf_krot", state.buf_krot)
    refuse_nonfinite("state.buf_v", state.buf_v)
    refuse_nonfinite("state.buf_dt", state.buf_dt)
    refuse_nonfinite("state.buf_sig", state.buf_sig)
    refuse_nonfinite("state.buf_adt", state.buf_adt)
    if state.pending:
        refuse_nonfinite("input_states.k", state.pend_k)
        refuse_nonfinite("input_states.v", state.pend_v)


def _zeros(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(0.0)
    return out^


struct Mamba3State(Copyable, Movable):
    """The carried state, DEVIATION 832's construction:

      1. `theta` [B, H, R]: the angle state after the last processed
         token (in [0, 2pi) by the S10 mod's construction);
      2. `h` [B, H, P, N]: the SSM state entering the LAST WORKING CHUNK
         (after all SEALED chunks -- 832(i); zeros, or the corrected
         `Input_States` h, before the first);
      3. the last working chunk's buffer: `buf_len` rows (1..Q after any
         call, 0 fresh) of ROTATED-UNSCALED q/k ([B, Q, H, N]), raw v
         ([B, Q, H, P]), post-softplus dt, sigma(trap) and ADT
         ([B, Q, H] each) -- contract section 5's list;
      4. the PENDING `Input_States` continuation pieces `pend_k`
         [B, H, N] / `pend_v` [B, H, P] (upstream's K_State/V_State),
         consumed by S22 on the next call's first token.

    One `buf_len` for the whole batch: every sequence in a launch
    advances together (varlen deferred by name, contract section 5)."""

    var b: Int
    var dims: Mamba3Dims
    var buf_len: Int
    var pending: Bool
    var theta: List[Float32]
    var h: List[Float32]
    var buf_qrot: List[Float32]
    var buf_krot: List[Float32]
    var buf_v: List[Float32]
    var buf_dt: List[Float32]
    var buf_sig: List[Float32]
    var buf_adt: List[Float32]
    var pend_k: List[Float32]
    var pend_v: List[Float32]

    def __init__(out self, b: Int, dims: Mamba3Dims):
        self.b = b
        self.dims = dims.copy()
        self.buf_len = 0
        self.pending = False
        var nh = dims.nheads
        self.theta = _zeros(b * nh * M3_NUM_ROPE_ANGLES)
        self.h = _zeros(b * nh * M3_HEADDIM * M3_D_STATE)
        self.buf_qrot = _zeros(b * M3_CHUNK_SIZE * nh * M3_D_STATE)
        self.buf_krot = _zeros(b * M3_CHUNK_SIZE * nh * M3_D_STATE)
        self.buf_v = _zeros(b * M3_CHUNK_SIZE * nh * M3_HEADDIM)
        self.buf_dt = _zeros(b * M3_CHUNK_SIZE * nh)
        self.buf_sig = _zeros(b * M3_CHUNK_SIZE * nh)
        self.buf_adt = _zeros(b * M3_CHUNK_SIZE * nh)
        self.pend_k = _zeros(b * nh * M3_D_STATE)
        self.pend_v = _zeros(b * nh * M3_HEADDIM)

    def set_input_states(
        mut self,
        theta_in: List[Float32],
        h_in: List[Float32],
        k_in: List[Float32],
        v_in: List[Float32],
    ) raises:
        """The upstream four-piece `Input_States` continuation (contract
        section 5 claim 2): legal only on a FRESH state. theta and h load
        directly; k/v are held PENDING for S22's correction, which needs
        the next call's first-token dt and sigma(trap) (fwd:367-371; the
        NORMATIVE ref's scalar-first association, :266-267, wins --
        seam S22). NOT claimed bit-equal to an unbroken prefill; the
        one-rounding-versus-two argument is the contract's."""
        if self.buf_len != 0:
            raise Error(
                "Mamba3State.set_input_states: the state is mid-sequence"
                " (buf_len = "
                + String(self.buf_len)
                + "); Input_States only has upstream meaning on a fresh"
                " state"
            )
        if len(theta_in) != len(self.theta) or len(h_in) != len(self.h):
            raise Error("Mamba3State.set_input_states: theta/h size mismatch")
        if len(k_in) != len(self.pend_k) or len(v_in) != len(self.pend_v):
            raise Error("Mamba3State.set_input_states: k/v size mismatch")
        for i in range(len(theta_in)):
            self.theta[i] = theta_in[i]
        for i in range(len(h_in)):
            self.h[i] = h_in[i]
        for i in range(len(k_in)):
            self.pend_k[i] = k_in[i]
        for i in range(len(v_in)):
            self.pend_v[i] = v_in[i]
        self.pending = True


struct Mamba3Stages(Movable):
    """Every recorded stage of one block call, contract section 7's card
    order, plus the compare-only `pass_states` and the after-call state
    copies the gates read. Token stages are token-major over the call's
    NEW tokens (M = B * l); chunk stages cover the WORKING chunks
    (DEVIATION 832(ii))."""

    var q0_at_entry: Int
    var t_work: Int
    var n_chunks: Int
    var norm_sumsq: List[Float32]  # [M]                    S1
    var norm_out: List[Float32]  # [M, d_model]             S2-S3
    var in_proj: List[Float32]  # [M, d_in_proj]            S4
    var a_out: List[Float32]  # [M, H]                      S5 (clamped)
    var dt_out: List[Float32]  # [M, H]                     S6
    var adt_out: List[Float32]  # [M, H]                    S7
    var trap_sigma: List[Float32]  # [M, H]                 S8
    var trap_scale: List[Float32]  # [M, H]                 S9 (NOT prefix-stable, 832(iii))
    var bcnorm_b: List[Float32]  # [M, N] (G = 1)           S21
    var bcnorm_c: List[Float32]  # [M, N]                   S21
    var angle_theta: List[Float32]  # [M, H, R]             S10 (post-mod)
    var rot_q: List[Float32]  # [M, H, N]                   S12-S13
    var rot_k: List[Float32]  # [M, H, N]                   S12-S13 (PRE-scale)
    var qkdot_out: List[Float32]  # [M, H]                  S14 (gamma-scaled)
    var kscale_out: List[Float32]  # [M, H, N]              S15 (NOT prefix-stable)
    var dacs_out: List[Float32]  # [B, H, C, Q]             mamba2 S11 inherited
    var seg_l: List[Float32]  # [B, C, H, Q, Q]             S16 decay; +0.0 ON and above diag
    var pass_states: List[Float32]  # [B, C, H, P, N] entering, compare-only
    var yintra_out: List[Float32]  # [M, H, P]              S16
    var ystate_out: List[Float32]  # [M, H, P]              S17
    var skip_out: List[Float32]  # [M, H, P]                S18 (+ DEV 833's combine)
    var gate_out: List[Float32]  # [M, H, P]                S19
    var out_proj: List[Float32]  # [M, d_model]             S4
    var residual_out: List[Float32]  # [M, d_model]         S23
    var h_last: List[Float32]  # [B, H, P, N]  section 5 report
    var k_last: List[Float32]  # [B, H, N]     section 5 report (PRE-scale)
    var v_last: List[Float32]  # [B, H, P]     section 5 report (raw)
    var theta_last: List[Float32]  # [B, H, R] section 5 report
    var state_h_after: List[Float32]  # [B, H, P, N] the SEALED boundary (copy)
    var state_theta_after: List[Float32]  # [B, H, R] (copy)

    def __init__(out self):
        self.q0_at_entry = 0
        self.t_work = 0
        self.n_chunks = 0
        self.norm_sumsq = List[Float32]()
        self.norm_out = List[Float32]()
        self.in_proj = List[Float32]()
        self.a_out = List[Float32]()
        self.dt_out = List[Float32]()
        self.adt_out = List[Float32]()
        self.trap_sigma = List[Float32]()
        self.trap_scale = List[Float32]()
        self.bcnorm_b = List[Float32]()
        self.bcnorm_c = List[Float32]()
        self.angle_theta = List[Float32]()
        self.rot_q = List[Float32]()
        self.rot_k = List[Float32]()
        self.qkdot_out = List[Float32]()
        self.kscale_out = List[Float32]()
        self.dacs_out = List[Float32]()
        self.seg_l = List[Float32]()
        self.pass_states = List[Float32]()
        self.yintra_out = List[Float32]()
        self.ystate_out = List[Float32]()
        self.skip_out = List[Float32]()
        self.gate_out = List[Float32]()
        self.out_proj = List[Float32]()
        self.residual_out = List[Float32]()
        self.h_last = List[Float32]()
        self.k_last = List[Float32]()
        self.v_last = List[Float32]()
        self.theta_last = List[Float32]()
        self.state_h_after = List[Float32]()
        self.state_theta_after = List[Float32]()


# ===========================================================================
# The block, contract section 2's order. ONE function, both paths
# (DEVIATION 831): prefill = fresh zero state; decode = the same call at
# l = 1 carrying the state.
# ===========================================================================


def mamba3_block_oracle(
    w: Mamba3Weights,
    x: List[Float32],  # [B, l, d_model] token-major -- the NEW tokens
    b: Int,
    l: Int,
    mut state: Mamba3State,
) raises -> Mamba3Stages:
    """One block call. Returns the section 7 card (+ pass_states and the
    state copies); mutates the carried state per DEVIATION 832."""
    m3_refuse_bad_inputs(w, x, state)
    var dims = w.dims.copy()
    var dm = dims.d_model
    var di = dims.d_inner
    var dip = dims.d_in_proj()
    var nh = dims.nheads
    comptime p_dim = M3_HEADDIM
    comptime n_state = M3_D_STATE
    comptime q = M3_CHUNK_SIZE
    comptime r_ang = M3_NUM_ROPE_ANGLES
    var m = b * l
    if len(x) != m * dm:
        raise Error(
            String("mamba3_block_oracle: ")
            + String(len(x))
            + " input values for B = "
            + String(b)
            + ", l = "
            + String(l)
            + ", d_model = "
            + String(dm)
        )
    var st = Mamba3Stages()
    st.q0_at_entry = state.buf_len

    # ---- S1-S3: block RMSNorm (block.py:51-53, :67 non-fused arm) --
    #      mamba2 S1-S3 VERBATIM (the mamba1 machinery).
    for t in range(m):
        var acc = Float32(0.0)
        for j in range(dm):
            var xj = ftz(x[t * dm + j])
            acc = ftz(identical_mul_add(xj, xj, acc))
        st.norm_sumsq.append(acc)
        var mean = ftz(identical_div(acc, Float32(dm)))
        var rstd = ftz(identical_rsqrt(ftz(mean + M3_RMS_EPS)))
        for j in range(dm):
            var inner = ftz(pinned_mul(ftz(x[t * dm + j]), rstd))
            st.norm_out.append(ftz(pinned_mul(ftz(w.norm_w[j]), inner)))

    # ---- S4: in_proj (mamba3.py:176; Linear, bias=False), gemm v1
    #      OP_NT, k = d_model. Columns z|x|B|C|dd_dt|dd_A|trap|angle.
    st.in_proj = gemm_oracle(st.norm_out, w.w_in, OP_NT, m, dip, dm)

    # ---- S5 (data-dependent A, clamped) + S6 (dt, NO clamp) per
    #      (token, head).
    var c_dt = dims.col_dt()
    var c_a = dims.col_a()
    for t in range(m):
        for hh in range(nh):
            st.a_out.append(m3_heavy_tail_a(st.in_proj[t * dip + c_a + hh]))
            var biased = ftz(
                ftz(st.in_proj[t * dip + c_dt + hh]) + ftz(w.dt_bias[hh])
            )
            st.dt_out.append(ftz(identical_softplus(biased)))

    # ---- S21: B/C RMSNorm over d_state per (token, group), G = 1 --
    #      the S1-S3 machinery at eps 1e-5, learned weight, NO gate, NO
    #      bias (rms_norm_ref:29-30 at z=None, group_size=None; the bias
    #      is S12's, added later, per head, in the core).
    var c_b = dims.col_b()
    var c_c = dims.col_c()
    for t in range(m):
        var accb = Float32(0.0)
        var accc = Float32(0.0)
        for n in range(n_state):
            var bj = ftz(st.in_proj[t * dip + c_b + n])
            accb = ftz(identical_mul_add(bj, bj, accb))
            var cj = ftz(st.in_proj[t * dip + c_c + n])
            accc = ftz(identical_mul_add(cj, cj, accc))
        var rstdb = ftz(
            identical_rsqrt(
                ftz(ftz(identical_div(accb, Float32(n_state))) + M3_RMS_EPS)
            )
        )
        var rstdc = ftz(
            identical_rsqrt(
                ftz(ftz(identical_div(accc, Float32(n_state))) + M3_RMS_EPS)
            )
        )
        for n in range(n_state):
            var innerb = ftz(
                pinned_mul(ftz(st.in_proj[t * dip + c_b + n]), rstdb)
            )
            st.bcnorm_b.append(ftz(pinned_mul(ftz(w.bnorm_w[n]), innerb)))
            var innerc = ftz(
                pinned_mul(ftz(st.in_proj[t * dip + c_c + n]), rstdc)
            )
            st.bcnorm_c.append(ftz(pinned_mul(ftz(w.cnorm_w[n]), innerc)))

    # ---- assemble the WORKING sequence (DEVIATION 832: the last working
    #      chunk's buffered rows ++ new rows). Copies, not seams. The
    #      working start is chunk-aligned by 832(i)'s invariant.
    var q0 = state.buf_len
    var t_work = q0 + l
    st.t_work = t_work
    var nc = (t_work + q - 1) // q
    if nc < 1:
        nc = 1
    st.n_chunks = nc
    var rotq_work = _zeros(b * t_work * nh * n_state)
    var rotk_work = _zeros(b * t_work * nh * n_state)
    var v_work = _zeros(b * t_work * nh * p_dim)
    var dt_work = _zeros(b * t_work * nh)
    var sig_work = _zeros(b * t_work * nh)
    var adt_work = _zeros(b * t_work * nh)
    var c_x = dims.col_x()
    for bb in range(b):
        for t in range(q0):
            for i in range(nh * n_state):
                rotq_work[(bb * t_work + t) * nh * n_state + i] = (
                    state.buf_qrot[(bb * q + t) * nh * n_state + i]
                )
                rotk_work[(bb * t_work + t) * nh * n_state + i] = (
                    state.buf_krot[(bb * q + t) * nh * n_state + i]
                )
            for i in range(nh * p_dim):
                v_work[(bb * t_work + t) * nh * p_dim + i] = state.buf_v[
                    (bb * q + t) * nh * p_dim + i
                ]
            for hh in range(nh):
                dt_work[(bb * t_work + t) * nh + hh] = state.buf_dt[
                    (bb * q + t) * nh + hh
                ]
                sig_work[(bb * t_work + t) * nh + hh] = state.buf_sig[
                    (bb * q + t) * nh + hh
                ]
                adt_work[(bb * t_work + t) * nh + hh] = state.buf_adt[
                    (bb * q + t) * nh + hh
                ]
        for li in range(l):
            var t = q0 + li
            var mm = bb * l + li
            for hh in range(nh):
                dt_work[(bb * t_work + t) * nh + hh] = st.dt_out[
                    mm * nh + hh
                ]
            for i in range(nh * p_dim):
                # v = the raw x split (no conv, no activation).
                v_work[(bb * t_work + t) * nh * p_dim + i] = st.in_proj[
                    mm * dip + c_x + i
                ]

    # ---- S7 (ADT) + S8 (sigma(trap)) for the NEW working rows.
    var c_trap = dims.col_trap()
    for bb in range(b):
        for li in range(l):
            var t = q0 + li
            var mm = bb * l + li
            for hh in range(nh):
                var adt = ftz(
                    pinned_mul(
                        ftz(st.a_out[mm * nh + hh]),
                        ftz(dt_work[(bb * t_work + t) * nh + hh]),
                    )
                )
                adt_work[(bb * t_work + t) * nh + hh] = adt
                st.adt_out.append(adt)
                var sg = ftz(
                    identical_sigmoid(
                        ftz(st.in_proj[mm * dip + c_trap + hh])
                    )
                )
                sig_work[(bb * t_work + t) * nh + hh] = sg
                st.trap_sigma.append(sg)

    # ---- S9: gamma, beta', scale over ALL working rows. The shifted
    #      operands past the sequence end are STRUCTURAL +0.0 (contract
    #      section 3: the last real token's beta' leg belongs to the NEXT
    #      call -- the trapezoid's seam, not a bug).
    var gamma_work = _zeros(b * t_work * nh)
    var betap_work = _zeros(b * t_work * nh)
    var scale_work = _zeros(b * t_work * nh)
    for bb in range(b):
        for t in range(t_work):
            for hh in range(nh):
                var g = ftz(
                    pinned_mul(
                        ftz(dt_work[(bb * t_work + t) * nh + hh]),
                        ftz(sig_work[(bb * t_work + t) * nh + hh]),
                    )
                )
                var bp = Float32(0.0)
                if t + 1 < t_work:
                    bp = ftz(
                        pinned_mul(
                            ftz(dt_work[(bb * t_work + t + 1) * nh + hh]),
                            ftz(
                                Float32(1.0)
                                - ftz(
                                    sig_work[
                                        (bb * t_work + t + 1) * nh + hh
                                    ]
                                )
                            ),
                        )
                    )
                gamma_work[(bb * t_work + t) * nh + hh] = g
                betap_work[(bb * t_work + t) * nh + hh] = bp
                scale_work[(bb * t_work + t) * nh + hh] = ftz(g + bp)
    for bb in range(b):
        for li in range(l):
            for hh in range(nh):
                st.trap_scale.append(
                    scale_work[(bb * t_work + q0 + li) * nh + hh]
                )

    # ---- S10: the angle recurrence, SERIAL per token, mod 2pi EVERY
    #      step (DEVIATION 829), theta_-1 = the carried angle state.
    var c_ang = dims.col_angle()
    st.angle_theta = _zeros(m * nh * r_ang)
    st.theta_last = _zeros(b * nh * r_ang)
    for bb in range(b):
        for hh in range(nh):
            for r in range(r_ang):
                var run = ftz(state.theta[(bb * nh + hh) * r_ang + r])
                for li in range(l):
                    var mm = bb * l + li
                    var a = ftz(
                        pinned_mul(
                            identical_tanh(
                                ftz(st.in_proj[mm * dip + c_ang + r])
                            ),
                            M3_PI,
                        )
                    )
                    var inc = ftz(
                        pinned_mul(
                            a,
                            ftz(
                                dt_work[
                                    (bb * t_work + q0 + li) * nh + hh
                                ]
                            ),
                        )
                    )
                    run = m3_mod_2pi(ftz(run + inc))
                    st.angle_theta[(mm * nh + hh) * r_ang + r] = run
                state.theta[(bb * nh + hh) * r_ang + r] = run
                st.theta_last[(bb * nh + hh) * r_ang + r] = run

    # ---- S12 (bias AFTER the norm) + S11/S13 (portable trig pair,
    #      interleaved pairs, UNFUSED; pairs >= R structurally unrotated,
    #      DEVIATION 828) for the NEW rows. B/C are broadcast to heads by
    #      COPY; the bias and the rotation are per head.
    for bb in range(b):
        for li in range(l):
            var t = q0 + li
            var mm = bb * l + li
            for hh in range(nh):
                for j in range(n_state // 2):
                    var e0 = 2 * j
                    var e1 = 2 * j + 1
                    var q0v = ftz(
                        ftz(st.bcnorm_c[mm * n_state + e0])
                        + ftz(w.c_bias[hh * n_state + e0])
                    )
                    var q1v = ftz(
                        ftz(st.bcnorm_c[mm * n_state + e1])
                        + ftz(w.c_bias[hh * n_state + e1])
                    )
                    var k0v = ftz(
                        ftz(st.bcnorm_b[mm * n_state + e0])
                        + ftz(w.b_bias[hh * n_state + e0])
                    )
                    var k1v = ftz(
                        ftz(st.bcnorm_b[mm * n_state + e1])
                        + ftz(w.b_bias[hh * n_state + e1])
                    )
                    var base = ((bb * t_work + t) * nh + hh) * n_state
                    if j < r_ang:
                        var th = ftz(
                            st.angle_theta[(mm * nh + hh) * r_ang + j]
                        )
                        var cv = ftz(portable_cosf(th))
                        var sv = ftz(portable_sinf(th))
                        rotq_work[base + e0] = ftz(
                            pinned_mul(q0v, cv) - pinned_mul(q1v, sv)
                        )
                        rotq_work[base + e1] = ftz(
                            pinned_mul(q0v, sv) + pinned_mul(q1v, cv)
                        )
                        rotk_work[base + e0] = ftz(
                            pinned_mul(k0v, cv) - pinned_mul(k1v, sv)
                        )
                        rotk_work[base + e1] = ftz(
                            pinned_mul(k0v, sv) + pinned_mul(k1v, cv)
                        )
                    else:
                        # STRUCTURAL identity: never computed trig
                        # (cos(+0.0)=1, sin(+0.0)=+0.0 agree bit for bit
                        # with the references' pad spellings, DEV 828).
                        rotq_work[base + e0] = q0v
                        rotq_work[base + e1] = q1v
                        rotk_work[base + e0] = k0v
                        rotk_work[base + e1] = k1v

    # rot.q / rot.k stages: the NEW-token slices (copies).
    for bb in range(b):
        for li in range(l):
            var t = q0 + li
            for i in range(nh * n_state):
                st.rot_q.append(
                    rotq_work[(bb * t_work + t) * nh * n_state + i]
                )
                st.rot_k.append(
                    rotk_work[(bb * t_work + t) * nh * n_state + i]
                )

    # ---- S14: pre-rotation QK dot, gemm v1 cell over n (k = 128, ONE
    #      serial ascending leaf), then times gamma (DEVIATION 830).
    for bb in range(b):
        for li in range(l):
            var mm = bb * l + li
            for hh in range(nh):
                var acc = Float32(0.0)
                for n in range(n_state):
                    var qv = ftz(
                        ftz(st.bcnorm_c[mm * n_state + n])
                        + ftz(w.c_bias[hh * n_state + n])
                    )
                    var kv = ftz(
                        ftz(st.bcnorm_b[mm * n_state + n])
                        + ftz(w.b_bias[hh * n_state + n])
                    )
                    acc = ftz(identical_mul_add(qv, kv, acc))
                st.qkdot_out.append(
                    ftz(
                        pinned_mul(
                            ftz(acc),
                            gamma_work[(bb * t_work + q0 + li) * nh + hh],
                        )
                    )
                )

    # ---- S15: K scaling over ALL working rows (the carried k-state and
    #      the k_last report are PRE-scale; fwd:337-344).
    var kscale_work = _zeros(b * t_work * nh * n_state)
    for bb in range(b):
        for t in range(t_work):
            for hh in range(nh):
                for n in range(n_state):
                    var idx = ((bb * t_work + t) * nh + hh) * n_state + n
                    kscale_work[idx] = ftz(
                        pinned_mul(
                            ftz(rotk_work[idx]),
                            scale_work[(bb * t_work + t) * nh + hh],
                        )
                    )
    for bb in range(b):
        for li in range(l):
            var t = q0 + li
            for i in range(nh * n_state):
                st.kscale_out.append(
                    kscale_work[(bb * t_work + t) * nh * n_state + i]
                )

    # ---- mamba2 S11 inherited: per-chunk serial ascending cumsum of
    #      ADT; the chunk boundary is a hard reset; padded positions COPY
    #      the last real value.
    st.dacs_out = _zeros(b * nh * nc * q)
    for bb in range(b):
        for hh in range(nh):
            for c in range(nc):
                var c0 = c * q
                var real = t_work - c0
                if real > q:
                    real = q
                var run = Float32(0.0)
                for i in range(q):
                    if i < real:
                        var v = ftz(
                            adt_work[(bb * t_work + c0 + i) * nh + hh]
                        )
                        if i == 0:
                            run = v
                        else:
                            run = ftz(run + v)
                    st.dacs_out[((bb * nh + hh) * nc + c) * q + i] = run

    # ---- S16's decay: L = exp(segsum), STRICT triangle -- +0.0 ON and
    #      above the diagonal, STRUCTURAL (DEVIATIONS 782 inherited +
    #      830's strict mask). Column-serial rebuild.
    st.seg_l = _zeros(b * nc * nh * q * q)
    for bb in range(b):
        for c in range(nc):
            var c0 = c * q
            var real = t_work - c0
            if real > q:
                real = q
            for hh in range(nh):
                var lbase = (((bb * nc + c) * nh + hh) * q) * q
                for j in range(q):
                    var acc = Float32(0.0)
                    for i in range(j + 1, q):
                        if i < real:
                            acc = ftz(
                                acc
                                + ftz(
                                    adt_work[
                                        (bb * t_work + c0 + i) * nh + hh
                                    ]
                                )
                            )
                        st.seg_l[lbase + i * q + j] = ftz(
                            identical_exp(acc)
                        )
                    # i <= j (diagonal included): stays the +0.0 fill --
                    # STRUCTURAL, never computed.

    # ---- S22: the pending Input_States correction, the NORMATIVE ref's
    #      scalar-first association (:266-267): c = pinned_mul(dt_1,
    #      ftz(1 - sigma_1)); t = pinned_mul(pinned_mul(v_st, k_st), c);
    #      h0 = ftz(h_in + t). dt_1/sigma_1 are the call's FIRST token's
    #      (a fresh call by set_input_states' guard, so q0 = 0).
    if state.pending:
        for bb in range(b):
            for hh in range(nh):
                var csc = ftz(
                    pinned_mul(
                        ftz(dt_work[(bb * t_work + 0) * nh + hh]),
                        ftz(
                            Float32(1.0)
                            - ftz(sig_work[(bb * t_work + 0) * nh + hh])
                        ),
                    )
                )
                for p in range(p_dim):
                    for n in range(n_state):
                        var idx = (
                            ((bb * nh + hh) * p_dim + p) * n_state + n
                        )
                        var tv = ftz(
                            pinned_mul(
                                ftz(
                                    pinned_mul(
                                        ftz(
                                            state.pend_v[
                                                (bb * nh + hh) * p_dim + p
                                            ]
                                        ),
                                        ftz(
                                            state.pend_k[
                                                (bb * nh + hh) * n_state
                                                + n
                                            ]
                                        ),
                                    )
                                ),
                                csc,
                            )
                        )
                        state.h[idx] = ftz(ftz(state.h[idx]) + tv)
        state.pending = False
        for i in range(len(state.pend_k)):
            state.pend_k[i] = 0.0
        for i in range(len(state.pend_v)):
            state.pend_v[i] = 0.0

    # ---- S20: the SERIAL inter-chunk pass. pass_states records the
    #      state ENTERING each working chunk; the carried h becomes the
    #      state entering the LAST working chunk (DEVIATION 832(i): the
    #      sealed boundary); h_last is the state after the FINAL padded
    #      chunk (the report).
    st.pass_states = _zeros(b * nc * nh * p_dim * n_state)
    st.h_last = _zeros(b * nh * p_dim * n_state)
    for bb in range(b):
        for hh in range(nh):
            var h_run = List[Float32]()
            for i in range(p_dim * n_state):
                h_run.append(
                    state.h[((bb * nh + hh) * p_dim) * n_state + i]
                )
            var h_sealed = h_run.copy()
            for c in range(nc):
                var pbase = (((bb * nc + c) * nh + hh) * p_dim) * n_state
                for i in range(p_dim * n_state):
                    st.pass_states[pbase + i] = h_run[i]
                if c == nc - 1:
                    for i in range(p_dim * n_state):
                        h_sealed[i] = h_run[i]
                # increment = (v ⊙ exp(da_cs_rev))^T . k_scaled, a gemm
                # v1 cell at k = Q = 64 (ONE leaf), output [P, N].
                var c0 = c * q
                var real = t_work - c0
                if real > q:
                    real = q
                var dl = ftz(
                    st.dacs_out[((bb * nh + hh) * nc + c) * q + (q - 1)]
                )
                var vs = _zeros(q * p_dim)
                var ks = _zeros(q * n_state)
                for i in range(real):
                    var drev = ftz(
                        dl
                        - ftz(
                            st.dacs_out[((bb * nh + hh) * nc + c) * q + i]
                        )
                    )
                    var e = ftz(identical_exp(drev))
                    for p in range(p_dim):
                        vs[i * p_dim + p] = ftz(
                            pinned_mul(
                                ftz(
                                    v_work[
                                        ((bb * t_work + c0 + i) * nh + hh)
                                        * p_dim
                                        + p
                                    ]
                                ),
                                e,
                            )
                        )
                    for n in range(n_state):
                        ks[i * n_state + n] = kscale_work[
                            ((bb * t_work + c0 + i) * nh + hh) * n_state
                            + n
                        ]
                    # padded rows: v is exact +0.0, so the fold sees
                    # exact zeros (contract section 3).
                var inc = gemm_oracle(vs, ks, OP_TN, p_dim, n_state, q)
                var scale_c = ftz(identical_exp(dl))
                for i in range(p_dim * n_state):
                    h_run[i] = ftz(
                        identical_mul_add(
                            scale_c, ftz(h_run[i]), ftz(inc[i])
                        )
                    )
            for i in range(p_dim * n_state):
                st.h_last[((bb * nh + hh) * p_dim) * n_state + i] = h_run[i]
                state.h[((bb * nh + hh) * p_dim) * n_state + i] = h_sealed[
                    i
                ]

    # ---- S16 (intra-chunk attention) + S17 (state read-out) + S18
    #      (diagonal + D, ONE add) + S19 (Z gate) for the NEW rows.
    #      DEVIATION 833: y = ftz(ystate + yintra), ystate the left
    #      operand (the kernel's accumulator order, fwd:406-418).
    st.yintra_out = _zeros(m * nh * p_dim)
    st.ystate_out = _zeros(m * nh * p_dim)
    st.skip_out = _zeros(m * nh * p_dim)
    st.gate_out = _zeros(m * nh * p_dim)
    var c_z = dims.col_z()
    for bb in range(b):
        for c in range(nc):
            var c0 = c * q
            var real = t_work - c0
            if real > q:
                real = q
            for hh in range(nh):
                # chunk matrices for this (b, c, h): rotated q rows,
                # scaled k rows, raw v rows (padded rows exact +0.0).
                var qmat = _zeros(q * n_state)
                var kmat = _zeros(q * n_state)
                var vmat = _zeros(q * p_dim)
                for i in range(real):
                    for n in range(n_state):
                        qmat[i * n_state + n] = rotq_work[
                            ((bb * t_work + c0 + i) * nh + hh) * n_state
                            + n
                        ]
                        kmat[i * n_state + n] = kscale_work[
                            ((bb * t_work + c0 + i) * nh + hh) * n_state
                            + n
                        ]
                    for p in range(p_dim):
                        vmat[i * p_dim + p] = v_work[
                            ((bb * t_work + c0 + i) * nh + hh) * p_dim + p
                        ]
                # s = q_rot . k_scaled^T, gemm v1 cells over n (k = 128);
                # only the strict triangle is USED (j < i); M's j >= i
                # entries are STRUCTURAL +0.0 (the -inf mask never
                # exists; the diagonal is DEVIATION 830's, moved to
                # S14/S18).
                var smat = gemm_oracle(qmat, kmat, OP_NT, q, q, n_state)
                var lbase = (((bb * nc + c) * nh + hh) * q) * q
                var m_mat = _zeros(q * q)
                for i in range(q):
                    for j in range(i):
                        m_mat[i * q + j] = ftz(
                            pinned_mul(
                                ftz(smat[i * q + j]),
                                ftz(st.seg_l[lbase + i * q + j]),
                            )
                        )
                var yint = gemm_oracle(m_mat, vmat, OP_NN, q, p_dim, q)
                # state read-out: (q_rot . h_entering^T) then * exp(da_cs)
                var h_in = _zeros(p_dim * n_state)
                var pbase = (((bb * nc + c) * nh + hh) * p_dim) * n_state
                for i in range(p_dim * n_state):
                    h_in[i] = st.pass_states[pbase + i]
                var ch = gemm_oracle(qmat, h_in, OP_NT, q, p_dim, n_state)
                for i in range(real):
                    var t = c0 + i
                    if t < q0:
                        continue  # buffered rows re-emit no outputs
                    var li = t - q0
                    var mm = bb * l + li
                    var e_i = ftz(
                        identical_exp(
                            ftz(
                                st.dacs_out[
                                    ((bb * nh + hh) * nc + c) * q + i
                                ]
                            )
                        )
                    )
                    for p in range(p_dim):
                        var yi = yint[i * p_dim + p]
                        var ys = ftz(
                            pinned_mul(ftz(ch[i * p_dim + p]), e_i)
                        )
                        st.yintra_out[(mm * nh + hh) * p_dim + p] = yi
                        st.ystate_out[(mm * nh + hh) * p_dim + p] = ys
                        # DEVIATION 833's ONE add, then S18's one add.
                        var y0 = ftz(ys + yi)
                        var tv = ftz(
                            ftz(w.d_skip[hh])
                            + st.qkdot_out[mm * nh + hh]
                        )
                        var pv = ftz(
                            pinned_mul(
                                tv,
                                ftz(
                                    v_work[
                                        ((bb * t_work + t) * nh + hh)
                                        * p_dim
                                        + p
                                    ]
                                ),
                            )
                        )
                        var sk = ftz(y0 + pv)
                        st.skip_out[(mm * nh + hh) * p_dim + p] = sk
                        # S19: the ONE-division silu (mamba1 744).
                        var zv = ftz(
                            st.in_proj[mm * dip + c_z + hh * p_dim + p]
                        )
                        st.gate_out[(mm * nh + hh) * p_dim + p] = ftz(
                            pinned_mul(ftz(sk), ftz(identical_silu(zv)))
                        )

    # ---- reports: k_last (post-bias post-rotation PRE-scale), v_last
    #      (raw) -- the last real working row (fwd wrapper :709-729).
    st.k_last = _zeros(b * nh * n_state)
    st.v_last = _zeros(b * nh * p_dim)
    for bb in range(b):
        for i in range(nh * n_state):
            st.k_last[bb * nh * n_state + i] = rotk_work[
                (bb * t_work + (t_work - 1)) * nh * n_state + i
            ]
        for i in range(nh * p_dim):
            st.v_last[bb * nh * p_dim + i] = v_work[
                (bb * t_work + (t_work - 1)) * nh * p_dim + i
            ]

    # ---- the buffer update (DEVIATION 832(i)): keep the LAST WORKING
    #      CHUNK's r = t_work - (C-1)*Q rows, r in [1, Q]. Copies.
    var r_keep = t_work - (nc - 1) * q
    for bb in range(b):
        for t in range(r_keep):
            var src = (nc - 1) * q + t
            for i in range(nh * n_state):
                state.buf_qrot[(bb * q + t) * nh * n_state + i] = (
                    rotq_work[(bb * t_work + src) * nh * n_state + i]
                )
                state.buf_krot[(bb * q + t) * nh * n_state + i] = (
                    rotk_work[(bb * t_work + src) * nh * n_state + i]
                )
            for i in range(nh * p_dim):
                state.buf_v[(bb * q + t) * nh * p_dim + i] = v_work[
                    (bb * t_work + src) * nh * p_dim + i
                ]
            for hh in range(nh):
                state.buf_dt[(bb * q + t) * nh + hh] = dt_work[
                    (bb * t_work + src) * nh + hh
                ]
                state.buf_sig[(bb * q + t) * nh + hh] = sig_work[
                    (bb * t_work + src) * nh + hh
                ]
                state.buf_adt[(bb * q + t) * nh + hh] = adt_work[
                    (bb * t_work + src) * nh + hh
                ]
    state.buf_len = r_keep

    # ---- S4: out_proj (mamba3.py:277), gemm v1 OP_NT, k = d_inner.
    #      The gate output IS the [M, d_inner] row (d = h*P + p, a copy).
    st.out_proj = gemm_oracle(st.gate_out, w.w_out, OP_NT, m, dm, di)

    # ---- S23: residual (block.py:52/:67), mamba2 S22 VERBATIM.
    for i in range(m * dm):
        st.residual_out.append(ftz(ftz(x[i]) + st.out_proj[i]))

    # Readable copies of the carried state for the gates.
    for i in range(len(state.h)):
        st.state_h_after.append(state.h[i])
    for i in range(len(state.theta)):
        st.state_theta_after.append(state.theta[i])
    return st^


# ===========================================================================
# The Float64 tolerance reference: the per-token trapezoidal recurrence
# (`mamba3_siso_step_ref`:99-146's alpha/beta/gamma algebra, which the
# chunked schedule factorizes exactly), plain double spellings. A
# TOLERANCE instrument; bitwise claims never touch it.
# ===========================================================================


struct Mamba3Ref64(Movable):
    var skip_out: List[Float64]  # [M, H, P]  y + D*v (pre-gate)
    var residual_out: List[Float64]  # [M, d_model]

    def __init__(out self):
        self.skip_out = List[Float64]()
        self.residual_out = List[Float64]()


def mamba3_block_ref64(
    w: Mamba3Weights,
    x: List[Float32],
    b: Int,
    l: Int,
    init_theta: List[Float32],  # [B, H, R], zeros for none
    init_h: List[Float32],  # [B, H, P, N]
    init_k: List[Float32],  # [B, H, N]
    init_v: List[Float32],  # [B, H, P]
) raises -> Mamba3Ref64:
    from std.math import cos, exp, floor, log, sin, sqrt, tanh

    var dims = w.dims.copy()
    var dm = dims.d_model
    var di = dims.d_inner
    var dip = dims.d_in_proj()
    var nh = dims.nheads
    comptime p_dim = M3_HEADDIM
    comptime n_state = M3_D_STATE
    comptime r_ang = M3_NUM_ROPE_ANGLES
    var m = b * l
    var out = Mamba3Ref64()
    var pi64 = Float64(3.141592653589793)
    var two_pi = 2.0 * pi64

    var norm = List[Float64]()
    for t in range(m):
        var acc = Float64(0.0)
        for j in range(dm):
            acc += Float64(x[t * dm + j]) * Float64(x[t * dm + j])
        var rstd = 1.0 / sqrt(acc / Float64(dm) + Float64(M3_RMS_EPS))
        for j in range(dm):
            norm.append(Float64(w.norm_w[j]) * (Float64(x[t * dm + j]) * rstd))

    var proj = List[Float64]()
    for t in range(m):
        for c in range(dip):
            var acc = Float64(0.0)
            for j in range(dm):
                acc += norm[t * dm + j] * Float64(w.w_in[c * dm + j])
            proj.append(acc)

    for _ in range(m * nh * p_dim):
        out.skip_out.append(0.0)

    var c_z = dims.col_z()
    var c_x = dims.col_x()
    var c_b = dims.col_b()
    var c_c = dims.col_c()
    var c_dt = dims.col_dt()
    var c_a = dims.col_a()
    var c_trap = dims.col_trap()
    var c_ang = dims.col_angle()

    for bb in range(b):
        for hh in range(nh):
            var s_state = List[Float64]()
            for i in range(p_dim * n_state):
                s_state.append(
                    Float64(init_h[((bb * nh + hh) * p_dim) * n_state + i])
                )
            var k_prev = List[Float64]()
            for n in range(n_state):
                k_prev.append(Float64(init_k[(bb * nh + hh) * n_state + n]))
            var v_prev = List[Float64]()
            for p in range(p_dim):
                v_prev.append(Float64(init_v[(bb * nh + hh) * p_dim + p]))
            var theta = List[Float64]()
            for r in range(r_ang):
                theta.append(
                    Float64(init_theta[(bb * nh + hh) * r_ang + r])
                )
            for li in range(l):
                var t = bb * l + li
                # dt, A, adt, sigma
                var dtv = proj[t * dip + c_dt + hh] + Float64(w.dt_bias[hh])
                if dtv <= 20.0:
                    dtv = log(1.0 + exp(dtv))
                var av = proj[t * dip + c_a + hh]
                var ht: Float64
                if av >= 0.0:
                    ht = 1.0 + av
                else:
                    ht = 1.0 / (1.0 - av)
                var a64 = -ht
                if a64 > -Float64(M3_A_FLOOR):
                    a64 = -Float64(M3_A_FLOOR)
                var adt = a64 * dtv
                var sg = 1.0 / (1.0 + exp(-proj[t * dip + c_trap + hh]))
                # normed B/C + bias, rotation by advanced theta
                var bn = 0.0
                var cn = 0.0
                for n in range(n_state):
                    bn += proj[t * dip + c_b + n] * proj[t * dip + c_b + n]
                    cn += proj[t * dip + c_c + n] * proj[t * dip + c_c + n]
                var rb = 1.0 / sqrt(bn / Float64(n_state) + Float64(M3_RMS_EPS))
                var rc = 1.0 / sqrt(cn / Float64(n_state) + Float64(M3_RMS_EPS))
                var kb = List[Float64]()
                var qb = List[Float64]()
                for n in range(n_state):
                    kb.append(
                        Float64(w.bnorm_w[n]) * (proj[t * dip + c_b + n] * rb)
                        + Float64(w.b_bias[hh * n_state + n])
                    )
                    qb.append(
                        Float64(w.cnorm_w[n]) * (proj[t * dip + c_c + n] * rc)
                        + Float64(w.c_bias[hh * n_state + n])
                    )
                for r in range(r_ang):
                    var ang = tanh(proj[t * dip + c_ang + r]) * pi64
                    var v = theta[r] + ang * dtv
                    theta[r] = v - two_pi * floor(v / two_pi)
                var k_rot = List[Float64]()
                var q_rot = List[Float64]()
                for n in range(n_state):
                    k_rot.append(kb[n])
                    q_rot.append(qb[n])
                for r in range(r_ang):
                    var cv = cos(theta[r])
                    var sv = sin(theta[r])
                    var e0 = 2 * r
                    var e1 = 2 * r + 1
                    q_rot[e0] = qb[e0] * cv - qb[e1] * sv
                    q_rot[e1] = qb[e0] * sv + qb[e1] * cv
                    k_rot[e0] = kb[e0] * cv - kb[e1] * sv
                    k_rot[e1] = kb[e0] * sv + kb[e1] * cv
                # the three-term trapezoidal update (step_ref :119-127)
                var alpha = exp(adt)
                var beta = (1.0 - sg) * dtv * alpha
                var gmm = sg * dtv
                for p in range(p_dim):
                    var vv = proj[t * dip + c_x + hh * p_dim + p]
                    for n in range(n_state):
                        var i = p * n_state + n
                        s_state[i] = (
                            alpha * s_state[i]
                            + beta * (k_prev[n] * v_prev[p])
                            + gmm * (k_rot[n] * vv)
                        )
                for p in range(p_dim):
                    var y = Float64(0.0)
                    for n in range(n_state):
                        y += s_state[p * n_state + n] * q_rot[n]
                    out.skip_out[(t * nh + hh) * p_dim + p] = (
                        y
                        + Float64(w.d_skip[hh])
                        * proj[t * dip + c_x + hh * p_dim + p]
                    )
                for n in range(n_state):
                    k_prev[n] = k_rot[n]
                for p in range(p_dim):
                    v_prev[p] = proj[t * dip + c_x + hh * p_dim + p]

    for t in range(m):
        # gate then out_proj then residual.
        var gate = List[Float64]()
        for j in range(di):
            var z = proj[t * dip + c_z + j]
            gate.append(out.skip_out[t * di + j] * (z / (1.0 + exp(-z))))
        for c in range(dm):
            var s = Float64(0.0)
            for j in range(di):
                s += gate[j] * Float64(w.w_out[c * di + j])
            out.residual_out.append(Float64(x[t * dm + c]) + s)
    return out^
