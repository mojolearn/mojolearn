# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The host Float32 oracle of one Mamba-3 (SISO) block under profile
`mojolearn.identical.mamba3.siso.fp32.v1`, and the Float64 tolerance
reference.

NO REFERENCE FILE -- the reference library ships no oracle; the ALGORITHM below
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
`mamba/impl/modules/mamba3.mojo` / `ops/mamba3_siso.mojo` is
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
prefill resumption via the buffer construction below; the reference
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
      `identical_mul(k, gamma)` and owe the beta' leg a second rounding
      later, which is exactly the split the contract proves unequal
      (section 5 claim 2); the sealed rule folds every sealed row ONCE at
      `identical_mul(k, ftz(gamma + beta'))`, which is what makes gate (d)'s
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
operand -- before S18's `ftz(y + identical_mul(ftz(D + qk_gamma), v))`.
Recorded as a deviation because the contract is silent and the reference
(unchunked) has no corresponding add at all.

DECODE IS PREFILL RESUMPTION (DEVIATION 831 + 832). ONE function serves
both paths: `mamba3_block_oracle` takes the carried state (theta, sealed
boundary h, the last working chunk's buffered rows), rebuilds the working
sequence, and runs the chunked schedule over it. A prefill is the same
call with a fresh zero state. The reference per-token step recurrence
(`mamba3_siso_step_ref`:119-127) is NOT here; it lives in the impl file
as the required-RED arm STEP_UPSTREAM_RECURRENCE.

The Float64 reference at the bottom is the per-token trapezoidal
recurrence (`mamba3_siso_step_ref`'s alpha/beta/gamma algebra, which the
chunked schedule factorizes exactly) in double precision. It is a
TOLERANCE instrument, never a bitwise one.
"""

from checks.numerics import (
    identical_mul,
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
from gemm.checks.gemm_oracle import (
    OP_NN,
    OP_NT,
    OP_TN,
    gemm_oracle,
    gemm_oracle_right_zero_padded,
)
from std.math import min

from std.time import perf_counter_ns

from std.os import getenv
from core.host_predict_threads import host_predict_task_count
from core.host_lanes import HostF32Ptr, host_block_timing_on, host_f32_uninit, host_row_tasks, host_tick
from core.host_parallel import host_parallelize
from gemm.host.gemm_host_rows import GHR_SERIAL_FMAS, gemm_host_rows, gemm_host_rows_right_zero_padded
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


def m3_neg_inf() -> Float32:
    """-inf for S5's `identical_clamp(., -inf, -A_floor)` lower bound
    (which never binds; the clamp primitive is used whole, DEVIATION
    788's requirement inherited)."""
    return -f32_from_bits(BITS_POS_INF)


def m3_mod_2pi(x: Float32) -> Float32:
    """DEVIATION 829's composed mod: `ftz(x - identical_mul(2pi,
    floor(identical_div(x, 2pi))))`, floor exact, 2pi the pinned bits.
    Not a primitive -- each side spells its own copy of this composition
    (the pinned_mul rule)."""
    from std.math import floor

    # ONE rounding, as every 0.8.19 build computed it: the old pin,
    # `fma(a, b, -0.0)`, folded to a contractable product that fused
    # into this subtract. Spelled out now that pinned_mul truly pins (lane/pinned-mul-contract-free).
    return ftz(
        identical_mul_add(-M3_TWO_PI, floor(identical_div(x, M3_TWO_PI)), x)
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


#: The inter-chunk pass splits each (batch, head) row into blocks of state
#: rows p only when the thread policy has more threads than rows (lane
#: neural-pass20): the blocks halve from the whole head down to
#: M3_INTER_PBLOCK_MIN rows until (batch x heads x blocks) reaches the
#: thread count. A schedule knob, it moves no bit (the per-cell chain and
#: the product's per-cell fold do not depend on which rows share a call);
#: a split row costs more (its right operand is gathered once per block),
#: so a host with threads <= rows keeps the whole-head rows.
#: MOJOLEARN_M3_INTER_PBLOCKS forces the block count (0: the rule).
comptime M3_INTER_PBLOCK_MIN = 8


def m3_inter_pblocks(bh_rows: Int, p_dim: Int) -> Int:
    var forced = 0
    try:
        forced = Int(String(getenv("MOJOLEARN_M3_INTER_PBLOCKS", "0")))
    except:
        forced = 0
    if forced > 0:
        var f = 1
        while f * 2 <= forced and p_dim // (f * 2) >= M3_INTER_PBLOCK_MIN:
            f *= 2
        return f
    var want = host_predict_task_count(1 << 30)
    var blocks = 1
    while blocks * bh_rows < want and p_dim // (blocks * 2) >= M3_INTER_PBLOCK_MIN:
        blocks *= 2
    return blocks


def _p(values: List[Float32]) -> HostF32Ptr:
    """The list's storage as the untracked mutable pointer the task closures
    capture (lane neural-pass9); the list outlives every task it is passed
    to, and tasks write disjoint indices."""
    return HostF32Ptr(unsafe_from_address=Int(values.unsafe_ptr()))


def _zeros(n: Int) -> List[Float32]:
    return List[Float32](length=n, fill=Float32(0.0))


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
         [B, H, N] / `pend_v` [B, H, P] (the reference's K_State/V_State),
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
        """The reference four-piece `Input_States` continuation (contract
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

    # =====================================================================
    # STAGES OVER HOST TASKS (lane neural-pass9). Every statement below is
    # the serial oracle's, in the serial oracle's order WITHIN a row; what
    # changed is which rows run on which task. A row is a token, a (batch,
    # head), a (batch, head, chunk) or a (batch, chunk, head), the unit the
    # contract already makes independent (the chunked scan's chunks share
    # nothing but `pass_states`, which the serial inter-chunk pass per head
    # writes before the chunk pass reads it). The stage lists are sized once
    # and written by index in the order the appends made them. Every task runs
    # in the calling thread's floating-point environment (`host_parallelize`,
    # DEVIATION 5900), and MOJOLEARN_CPU_THREADS=1 runs every stage as one
    # task on the calling thread: tools/host_threads_ab_check.py holds the two
    # equal byte for byte. The small per-chunk gemm calls inside a task stay
    # single-task because their work is under GHR_SERIAL_FMAS; a shape whose
    # chunk products would split runs those two stages on one task.
    # =====================================================================
    var p_dim_v = p_dim
    var n_state_v = n_state
    var q_v = q
    var r_ang_v = r_ang
    var xp = _p(x)
    var normw_p = _p(w.norm_w)
    var dtb_p = _p(w.dt_bias)
    var bnormw_p = _p(w.bnorm_w)
    var cnormw_p = _p(w.cnorm_w)
    var bbias_p = _p(w.b_bias)
    var cbias_p = _p(w.c_bias)
    var dskip_p = _p(w.d_skip)
    var hton = host_block_timing_on()
    var htk = Int(perf_counter_ns())

    # ---- S1-S3: block RMSNorm (block.py:51-53, :67 non-fused arm), tokens --
    st.norm_sumsq = host_f32_uninit(m)
    st.norm_out = host_f32_uninit(m * dm)
    var nsq_p = _p(st.norm_sumsq)
    var nout_p = _p(st.norm_out)
    var ttasks = host_row_tasks(m, 6 * dm)
    var tchunk = (m + ttasks - 1) // ttasks
    def _norm_rows(task: Int) {imm xp, imm normw_p, imm nsq_p, imm nout_p, imm m, imm dm, imm tchunk}:
        for t in range(task * tchunk, min((task + 1) * tchunk, m)):
            var acc = Float32(0.0)
            for j in range(dm):
                var xj = ftz(xp.unsafe_load(t * dm + j))
                acc = ftz(identical_mul_add(xj, xj, acc))
            nsq_p.unsafe_store(t, acc)
            var mean = ftz(identical_div(acc, Float32(dm)))
            var rstd = ftz(identical_rsqrt(ftz(mean + M3_RMS_EPS)))
            for j in range(dm):
                var inner = ftz(identical_mul(ftz(xp.unsafe_load(t * dm + j)), rstd))
                nout_p.unsafe_store(t * dm + j, ftz(identical_mul(ftz(normw_p.unsafe_load(j)), inner)))
    if ttasks <= 1:
        _norm_rows(0)
    else:
        host_parallelize(_norm_rows, ttasks)

    host_tick(hton, htk, "m3.norm")
    # ---- S4: in_proj (mamba3.py:176; Linear, bias=False), gemm v1 ---------
    st.in_proj = gemm_host_rows(st.norm_out, w.w_in, OP_NT, m, dip, dm)
    var ip_p = _p(st.in_proj)

    host_tick(hton, htk, "m3.in_proj")
    # ---- S5 (data-dependent A, clamped) + S6 (dt, NO clamp) per (token, head)
    var c_dt = dims.col_dt()
    var c_a = dims.col_a()
    st.a_out = host_f32_uninit(m * nh)
    st.dt_out = host_f32_uninit(m * nh)
    var aout_p = _p(st.a_out)
    var dtout_p = _p(st.dt_out)
    var htasks = host_row_tasks(m, 8 * nh)
    var hchunk = (m + htasks - 1) // htasks
    def _a_dt_rows(task: Int) {imm ip_p, imm dtb_p, imm aout_p, imm dtout_p, imm m, imm nh, imm dip, imm c_dt, imm c_a, imm hchunk}:
        for t in range(task * hchunk, min((task + 1) * hchunk, m)):
            for hh in range(nh):
                aout_p.unsafe_store(t * nh + hh, m3_heavy_tail_a(ip_p.unsafe_load(t * dip + c_a + hh)))
                var biased = ftz(
                    ftz(ip_p.unsafe_load(t * dip + c_dt + hh)) + ftz(dtb_p.unsafe_load(hh))
                )
                dtout_p.unsafe_store(t * nh + hh, ftz(identical_softplus(biased)))
    if htasks <= 1:
        _a_dt_rows(0)
    else:
        host_parallelize(_a_dt_rows, htasks)

    host_tick(hton, htk, "m3.a_dt")
    # ---- S21: B/C RMSNorm over d_state per (token, group), G = 1, tokens --
    var c_b = dims.col_b()
    var c_c = dims.col_c()
    st.bcnorm_b = host_f32_uninit(m * n_state)
    st.bcnorm_c = host_f32_uninit(m * n_state)
    var bcb_p = _p(st.bcnorm_b)
    var bcc_p = _p(st.bcnorm_c)
    var ntasks = host_row_tasks(m, 8 * n_state)
    var nchunk = (m + ntasks - 1) // ntasks
    def _bc_norm_rows(task: Int) {imm ip_p, imm bnormw_p, imm cnormw_p, imm bcb_p, imm bcc_p, imm m, imm dip, imm c_b, imm c_c, imm n_state_v, imm nchunk}:
        for t in range(task * nchunk, min((task + 1) * nchunk, m)):
            var accb = Float32(0.0)
            var accc = Float32(0.0)
            for n in range(n_state_v):
                var bj = ftz(ip_p.unsafe_load(t * dip + c_b + n))
                accb = ftz(identical_mul_add(bj, bj, accb))
                var cj = ftz(ip_p.unsafe_load(t * dip + c_c + n))
                accc = ftz(identical_mul_add(cj, cj, accc))
            var rstdb = ftz(
                identical_rsqrt(
                    ftz(ftz(identical_div(accb, Float32(n_state_v))) + M3_RMS_EPS)
                )
            )
            var rstdc = ftz(
                identical_rsqrt(
                    ftz(ftz(identical_div(accc, Float32(n_state_v))) + M3_RMS_EPS)
                )
            )
            for n in range(n_state_v):
                var innerb = ftz(
                    identical_mul(ftz(ip_p.unsafe_load(t * dip + c_b + n)), rstdb)
                )
                bcb_p.unsafe_store(t * n_state_v + n, ftz(identical_mul(ftz(bnormw_p.unsafe_load(n)), innerb)))
                var innerc = ftz(
                    identical_mul(ftz(ip_p.unsafe_load(t * dip + c_c + n)), rstdc)
                )
                bcc_p.unsafe_store(t * n_state_v + n, ftz(identical_mul(ftz(cnormw_p.unsafe_load(n)), innerc)))
    if ntasks <= 1:
        _bc_norm_rows(0)
    else:
        host_parallelize(_bc_norm_rows, ntasks)

    host_tick(hton, htk, "m3.bc_norm")
    # ---- assemble the WORKING sequence (DEVIATION 832: the last working
    # chunk's rows first, then this call's tokens) --------------------------
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
    var rotq_p = _p(rotq_work)
    var rotk_p = _p(rotk_work)
    var vw_p = _p(v_work)
    var dtw_p = _p(dt_work)
    var sigw_p = _p(sig_work)
    var adtw_p = _p(adt_work)
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
    # this call's tokens: (batch, token) rows over tasks
    var mtasks = host_row_tasks(m, 2 * nh * p_dim)
    var mchunk = (m + mtasks - 1) // mtasks
    def _new_token_rows(task: Int) {imm ip_p, imm dtout_p, imm vw_p, imm dtw_p, imm m, imm l, imm nh, imm dip, imm c_x, imm q0, imm t_work, imm p_dim_v, imm mchunk}:
        for mm in range(task * mchunk, min((task + 1) * mchunk, m)):
            var bb = mm // l
            var li = mm % l
            var t = q0 + li
            for hh in range(nh):
                dtw_p.unsafe_store((bb * t_work + t) * nh + hh, dtout_p.unsafe_load(mm * nh + hh))
            for i in range(nh * p_dim_v):
                vw_p.unsafe_store((bb * t_work + t) * nh * p_dim_v + i, ip_p.unsafe_load(mm * dip + c_x + i))
    if mtasks <= 1:
        _new_token_rows(0)
    else:
        host_parallelize(_new_token_rows, mtasks)

    host_tick(hton, htk, "m3.working_seq")
    # ---- S7 (ADT) + S8 (sigma(trap)) for the NEW working rows, tokens -----
    var c_trap = dims.col_trap()
    st.adt_out = host_f32_uninit(m * nh)
    st.trap_sigma = host_f32_uninit(m * nh)
    var adtout_p = _p(st.adt_out)
    var tsig_p = _p(st.trap_sigma)
    def _adt_sig_rows(task: Int) {imm ip_p, imm aout_p, imm dtw_p, imm adtw_p, imm sigw_p, imm adtout_p, imm tsig_p, imm m, imm l, imm nh, imm dip, imm c_trap, imm q0, imm t_work, imm hchunk}:
        for mm in range(task * hchunk, min((task + 1) * hchunk, m)):
            var bb = mm // l
            var li = mm % l
            var t = q0 + li
            for hh in range(nh):
                var adt = ftz(
                    identical_mul(
                        ftz(aout_p.unsafe_load(mm * nh + hh)),
                        ftz(dtw_p.unsafe_load((bb * t_work + t) * nh + hh)),
                    )
                )
                adtw_p.unsafe_store((bb * t_work + t) * nh + hh, adt)
                adtout_p.unsafe_store(mm * nh + hh, adt)
                var sg = ftz(
                    identical_sigmoid(
                        ftz(ip_p.unsafe_load(mm * dip + c_trap + hh))
                    )
                )
                sigw_p.unsafe_store((bb * t_work + t) * nh + hh, sg)
                tsig_p.unsafe_store(mm * nh + hh, sg)
    if htasks <= 1:
        _adt_sig_rows(0)
    else:
        host_parallelize(_adt_sig_rows, htasks)

    host_tick(hton, htk, "m3.adt_sigma")
    # ---- S9: gamma, beta', scale over ALL working rows, (batch, t) rows ----
    var gamma_work = _zeros(b * t_work * nh)
    var betap_work = _zeros(b * t_work * nh)
    var scale_work = _zeros(b * t_work * nh)
    var gam_p = _p(gamma_work)
    var bet_p = _p(betap_work)
    var scl_p = _p(scale_work)
    var wrows = b * t_work
    var wtasks = host_row_tasks(wrows, 6 * nh)
    var wchunk = (wrows + wtasks - 1) // wtasks
    def _gamma_rows(task: Int) {imm dtw_p, imm sigw_p, imm gam_p, imm bet_p, imm scl_p, imm wrows, imm t_work, imm nh, imm wchunk}:
        for r in range(task * wchunk, min((task + 1) * wchunk, wrows)):
            var bb = r // t_work
            var t = r % t_work
            for hh in range(nh):
                var g = ftz(
                    identical_mul(
                        ftz(dtw_p.unsafe_load((bb * t_work + t) * nh + hh)),
                        ftz(sigw_p.unsafe_load((bb * t_work + t) * nh + hh)),
                    )
                )
                var bp = Float32(0.0)
                if t + 1 < t_work:
                    bp = ftz(
                        identical_mul(
                            ftz(dtw_p.unsafe_load((bb * t_work + t + 1) * nh + hh)),
                            ftz(
                                Float32(1.0)
                                - ftz(
                                    sigw_p.unsafe_load((bb * t_work + t + 1) * nh + hh)
                                )
                            ),
                        )
                    )
                gam_p.unsafe_store((bb * t_work + t) * nh + hh, g)
                bet_p.unsafe_store((bb * t_work + t) * nh + hh, bp)
                scl_p.unsafe_store((bb * t_work + t) * nh + hh, ftz(g + bp))
    if wtasks <= 1:
        _gamma_rows(0)
    else:
        host_parallelize(_gamma_rows, wtasks)
    st.trap_scale = host_f32_uninit(m * nh)
    for bb in range(b):
        for li in range(l):
            for hh in range(nh):
                st.trap_scale[(bb * l + li) * nh + hh] = scale_work[(bb * t_work + q0 + li) * nh + hh]

    host_tick(hton, htk, "m3.gamma_scale")
    # ---- S10: the angle recurrence, SERIAL per token, mod 2pi EVERY step;
    # (batch, head) rows over tasks, each row its own r_ang chains ----------
    var c_ang = dims.col_angle()
    st.angle_theta = _zeros(m * nh * r_ang)
    st.theta_last = _zeros(b * nh * r_ang)
    var ang_p = _p(st.angle_theta)
    var thl_p = _p(st.theta_last)
    var stheta_p = _p(state.theta)
    var bh_rows = b * nh
    var atasks = host_row_tasks(bh_rows, 24 * l * r_ang)
    var achunk = (bh_rows + atasks - 1) // atasks
    def _angle_rows(task: Int) {imm ip_p, imm dtw_p, imm ang_p, imm thl_p, imm stheta_p, imm bh_rows, imm l, imm nh, imm dip, imm c_ang, imm q0, imm t_work, imm r_ang_v, imm achunk}:
        for r0 in range(task * achunk, min((task + 1) * achunk, bh_rows)):
            var bb = r0 // nh
            var hh = r0 % nh
            for r in range(r_ang_v):
                var run = ftz(stheta_p.unsafe_load((bb * nh + hh) * r_ang_v + r))
                for li in range(l):
                    var mm = bb * l + li
                    var a = ftz(
                        identical_mul(
                            identical_tanh(
                                ftz(ip_p.unsafe_load(mm * dip + c_ang + r))
                            ),
                            M3_PI,
                        )
                    )
                    var inc = ftz(
                        identical_mul(
                            a,
                            ftz(
                                dtw_p.unsafe_load((bb * t_work + q0 + li) * nh + hh)
                            ),
                        )
                    )
                    run = m3_mod_2pi(ftz(run + inc))
                    ang_p.unsafe_store((mm * nh + hh) * r_ang_v + r, run)
                stheta_p.unsafe_store((bb * nh + hh) * r_ang_v + r, run)
                thl_p.unsafe_store((bb * nh + hh) * r_ang_v + r, run)
    if atasks <= 1:
        _angle_rows(0)
    else:
        host_parallelize(_angle_rows, atasks)

    host_tick(hton, htk, "m3.angle")
    # ---- S12 (bias AFTER the norm) + S11/S13 (portable trig pair, the
    # rotation), tokens over tasks ------------------------------------------
    var rtasks = host_row_tasks(m, 24 * nh * n_state)
    var rchunk = (m + rtasks - 1) // rtasks
    def _rotation_rows(task: Int) {imm bcb_p, imm bcc_p, imm bbias_p, imm cbias_p, imm ang_p, imm rotq_p, imm rotk_p, imm m, imm l, imm nh, imm q0, imm t_work, imm n_state_v, imm r_ang_v, imm rchunk}:
        for mm in range(task * rchunk, min((task + 1) * rchunk, m)):
            var bb = mm // l
            var li = mm % l
            var t = q0 + li
            for hh in range(nh):
                for j in range(n_state_v // 2):
                    var e0 = 2 * j
                    var e1 = 2 * j + 1
                    var q0v = ftz(
                        ftz(bcc_p.unsafe_load(mm * n_state_v + e0))
                        + ftz(cbias_p.unsafe_load(hh * n_state_v + e0))
                    )
                    var q1v = ftz(
                        ftz(bcc_p.unsafe_load(mm * n_state_v + e1))
                        + ftz(cbias_p.unsafe_load(hh * n_state_v + e1))
                    )
                    var k0v = ftz(
                        ftz(bcb_p.unsafe_load(mm * n_state_v + e0))
                        + ftz(bbias_p.unsafe_load(hh * n_state_v + e0))
                    )
                    var k1v = ftz(
                        ftz(bcb_p.unsafe_load(mm * n_state_v + e1))
                        + ftz(bbias_p.unsafe_load(hh * n_state_v + e1))
                    )
                    var base = ((bb * t_work + t) * nh + hh) * n_state_v
                    if j < r_ang_v:
                        var th = ftz(
                            ang_p.unsafe_load((mm * nh + hh) * r_ang_v + j)
                        )
                        var cv = ftz(portable_cosf(th))
                        var sv = ftz(portable_sinf(th))
                        rotq_p.unsafe_store(base + e0, ftz(
                            ftz(identical_mul(q0v, cv))
                            - ftz(identical_mul(q1v, sv))
                        ))
                        rotq_p.unsafe_store(base + e1, ftz(
                            ftz(identical_mul(q0v, sv))
                            + ftz(identical_mul(q1v, cv))
                        ))
                        rotk_p.unsafe_store(base + e0, ftz(
                            ftz(identical_mul(k0v, cv))
                            - ftz(identical_mul(k1v, sv))
                        ))
                        rotk_p.unsafe_store(base + e1, ftz(
                            ftz(identical_mul(k0v, sv))
                            + ftz(identical_mul(k1v, cv))
                        ))
                    else:
                        rotq_p.unsafe_store(base + e0, q0v)
                        rotq_p.unsafe_store(base + e1, q1v)
                        rotk_p.unsafe_store(base + e0, k0v)
                        rotk_p.unsafe_store(base + e1, k1v)
    if rtasks <= 1:
        _rotation_rows(0)
    else:
        host_parallelize(_rotation_rows, rtasks)
    st.rot_q = host_f32_uninit(m * nh * n_state)
    st.rot_k = host_f32_uninit(m * nh * n_state)
    var rotq_out_p = _p(st.rot_q)
    var rotk_out_p = _p(st.rot_k)
    def _rot_copy_rows(task: Int) {imm rotq_p, imm rotk_p, imm rotq_out_p, imm rotk_out_p, imm m, imm l, imm nh, imm q0, imm t_work, imm n_state_v, imm mchunk}:
        for mm in range(task * mchunk, min((task + 1) * mchunk, m)):
            var bb = mm // l
            var li = mm % l
            var t = q0 + li
            for i in range(nh * n_state_v):
                rotq_out_p.unsafe_store(mm * nh * n_state_v + i, rotq_p.unsafe_load((bb * t_work + t) * nh * n_state_v + i))
                rotk_out_p.unsafe_store(mm * nh * n_state_v + i, rotk_p.unsafe_load((bb * t_work + t) * nh * n_state_v + i))
    if mtasks <= 1:
        _rot_copy_rows(0)
    else:
        host_parallelize(_rot_copy_rows, mtasks)

    host_tick(hton, htk, "m3.rotation")
    # ---- S14: pre-rotation QK dot, gemm v1 cell over n (k = 128, ONE
    # leaf), gamma-scaled; tokens over tasks ---------------------------------
    st.qkdot_out = host_f32_uninit(m * nh)
    var qkd_p = _p(st.qkdot_out)
    def _qkdot_rows(task: Int) {imm bcb_p, imm bcc_p, imm bbias_p, imm cbias_p, imm gam_p, imm qkd_p, imm m, imm l, imm nh, imm q0, imm t_work, imm n_state_v, imm rchunk}:
        for mm in range(task * rchunk, min((task + 1) * rchunk, m)):
            var bb = mm // l
            var li = mm % l
            for hh in range(nh):
                var acc = Float32(0.0)
                for n in range(n_state_v):
                    var qv = ftz(
                        ftz(bcc_p.unsafe_load(mm * n_state_v + n))
                        + ftz(cbias_p.unsafe_load(hh * n_state_v + n))
                    )
                    var kv = ftz(
                        ftz(bcb_p.unsafe_load(mm * n_state_v + n))
                        + ftz(bbias_p.unsafe_load(hh * n_state_v + n))
                    )
                    acc = ftz(identical_mul_add(qv, kv, acc))
                qkd_p.unsafe_store(mm * nh + hh, ftz(
                    identical_mul(
                        ftz(acc),
                        gam_p.unsafe_load((bb * t_work + q0 + li) * nh + hh),
                    )
                ))
    if rtasks <= 1:
        _qkdot_rows(0)
    else:
        host_parallelize(_qkdot_rows, rtasks)

    host_tick(hton, htk, "m3.qkdot")
    # ---- S15: K scaling over ALL working rows, (batch, t) rows over tasks -
    var kscale_work = _zeros(b * t_work * nh * n_state)
    var ksw_p = _p(kscale_work)
    def _kscale_rows(task: Int) {imm rotk_p, imm scl_p, imm ksw_p, imm wrows, imm t_work, imm nh, imm n_state_v, imm wchunk}:
        for r in range(task * wchunk, min((task + 1) * wchunk, wrows)):
            var bb = r // t_work
            var t = r % t_work
            for hh in range(nh):
                for n in range(n_state_v):
                    var idx = ((bb * t_work + t) * nh + hh) * n_state_v + n
                    ksw_p.unsafe_store(idx, ftz(
                        identical_mul(
                            ftz(rotk_p.unsafe_load(idx)),
                            scl_p.unsafe_load((bb * t_work + t) * nh + hh),
                        )
                    ))
    if wtasks <= 1:
        _kscale_rows(0)
    else:
        host_parallelize(_kscale_rows, wtasks)
    st.kscale_out = host_f32_uninit(m * nh * n_state)
    var kso_p = _p(st.kscale_out)
    def _kscale_copy_rows(task: Int) {imm ksw_p, imm kso_p, imm m, imm l, imm nh, imm q0, imm t_work, imm n_state_v, imm mchunk}:
        for mm in range(task * mchunk, min((task + 1) * mchunk, m)):
            var bb = mm // l
            var li = mm % l
            var t = q0 + li
            for i in range(nh * n_state_v):
                kso_p.unsafe_store(mm * nh * n_state_v + i, ksw_p.unsafe_load((bb * t_work + t) * nh * n_state_v + i))
    if mtasks <= 1:
        _kscale_copy_rows(0)
    else:
        host_parallelize(_kscale_copy_rows, mtasks)

    host_tick(hton, htk, "m3.kscale")
    # ---- mamba2 S11 inherited: per-chunk serial ascending cumsum of ADT,
    # (batch, head, chunk) rows over tasks -----------------------------------
    st.dacs_out = _zeros(b * nh * nc * q)
    var dacs_p = _p(st.dacs_out)
    var hc_rows = b * nh * nc
    var ctasks = host_row_tasks(hc_rows, 2 * q)
    var cchunk = (hc_rows + ctasks - 1) // ctasks
    def _cumsum_rows(task: Int) {imm adtw_p, imm dacs_p, imm hc_rows, imm nh, imm nc, imm t_work, imm q_v, imm cchunk}:
        for r in range(task * cchunk, min((task + 1) * cchunk, hc_rows)):
            var bb = r // (nh * nc)
            var hh = (r // nc) % nh
            var c = r % nc
            var c0 = c * q_v
            var real = t_work - c0
            if real > q_v:
                real = q_v
            var run = Float32(0.0)
            for i in range(q_v):
                if i < real:
                    var v = ftz(
                        adtw_p.unsafe_load((bb * t_work + c0 + i) * nh + hh)
                    )
                    if i == 0:
                        run = v
                    else:
                        run = ftz(run + v)
                dacs_p.unsafe_store(((bb * nh + hh) * nc + c) * q_v + i, run)
    if ctasks <= 1:
        _cumsum_rows(0)
    else:
        host_parallelize(_cumsum_rows, ctasks)

    host_tick(hton, htk, "m3.cumsum")
    # ---- S16's decay: L = exp(segsum), STRICT triangle -- +0.0 ON and
    # above the diagonal; (batch, chunk, head) rows over tasks -------------
    st.seg_l = _zeros(b * nc * nh * q * q)
    var segl_p = _p(st.seg_l)
    var ch_rows = b * nc * nh
    var dtasks = host_row_tasks(ch_rows, 3 * q * q)
    var dchunk = (ch_rows + dtasks - 1) // dtasks
    def _decay_rows(task: Int) {imm adtw_p, imm segl_p, imm ch_rows, imm nh, imm nc, imm t_work, imm q_v, imm dchunk}:
        for r in range(task * dchunk, min((task + 1) * dchunk, ch_rows)):
            var bb = r // (nc * nh)
            var c = (r // nh) % nc
            var hh = r % nh
            var c0 = c * q_v
            var real = t_work - c0
            if real > q_v:
                real = q_v
            var lbase = (((bb * nc + c) * nh + hh) * q_v) * q_v
            for j in range(q_v):
                var acc = Float32(0.0)
                for i in range(j + 1, q_v):
                    if i < real:
                        acc = ftz(
                            acc
                            + ftz(
                                adtw_p.unsafe_load(
                                    (bb * t_work + c0 + i) * nh + hh
                                )
                            )
                        )
                    segl_p.unsafe_store(lbase + i * q_v + j, ftz(
                        identical_exp(acc)
                    ))
    if dtasks <= 1:
        _decay_rows(0)
    else:
        host_parallelize(_decay_rows, dtasks)

    host_tick(hton, htk, "m3.decay")
    # ---- S22: the pending Input_States correction, the NORMATIVE ref's
    # `set_input_states` arm (unchanged, on the calling thread) -------------
    if state.pending:
        for bb in range(b):
            for hh in range(nh):
                var csc = ftz(
                    identical_mul(
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
                            identical_mul(
                                ftz(
                                    identical_mul(
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

    # The per-chunk gemm calls inside the two task stages below stay
    # single-task only while their work is under GHR_SERIAL_FMAS (no task
    # starts a parallel region); otherwise those stages run on one task.
    var chunk_gemms_serial = (p_dim * n_state * q < GHR_SERIAL_FMAS
                              and q * q * n_state < GHR_SERIAL_FMAS
                              and q * p_dim * q < GHR_SERIAL_FMAS
                              and q * p_dim * n_state < GHR_SERIAL_FMAS)

    host_tick(hton, htk, "m3.pending")
    # ---- S20: the SERIAL inter-chunk pass. pass_states records the
    # entering state of every chunk; (batch, head) rows over tasks ---------
    st.pass_states = _zeros(b * nc * nh * p_dim * n_state)
    st.h_last = _zeros(b * nh * p_dim * n_state)
    var pass_p = _p(st.pass_states)
    var hlast_p = _p(st.h_last)
    var sh_p = _p(state.h)
    var vwork_p = _p(v_work)
    # Lane neural-pass20: the recurrence of state cell (p, n) over the
    # chunks depends on no other cell (h = exp(dl) h + inc[p, n], inc the
    # product's cell, whose fold over the chunk's rows is the gemm's own
    # and does not depend on which rows of the left operand share the
    # call: gemm_host_rows' module note, dense_check), so the rows of the
    # split are (batch, head, block of state rows p), `m3_inter_pblocks`
    # blocks a head, and each task builds the left operand of its own rows
    # only. At the board's CPU cell (12 heads) a 64-thread host gets 96
    # rows where it had 12; a 10-thread host keeps the 12.
    var pblocks = m3_inter_pblocks(bh_rows, p_dim)
    var pblock = (p_dim + pblocks - 1) // pblocks
    var ic_rows = bh_rows * pblocks
    var ptasks = host_row_tasks(ic_rows, 4 * nc * q * pblock * n_state)
    if not chunk_gemms_serial:
        ptasks = 1
    var pchunk = (ic_rows + ptasks - 1) // ptasks
    var err_flag = _zeros(1)
    var err_p = _p(err_flag)
    def _inter_chunk_rows(task: Int) {imm dacs_p, imm vwork_p, imm ksw_p, imm pass_p, imm hlast_p, imm sh_p, imm err_p, imm ic_rows, imm pblocks, imm pblock, imm nh, imm nc, imm t_work, imm q_v, imm p_dim_v, imm n_state_v, imm pchunk}:
        for r0 in range(task * pchunk, min((task + 1) * pchunk, ic_rows)):
            var bh = r0 // pblocks
            var pb = r0 % pblocks
            var bb = bh // nh
            var hh = bh % nh
            var p0 = pb * pblock
            var p1 = min(p0 + pblock, p_dim_v)
            var pw = p1 - p0
            var cells = pw * n_state_v
            var sbase = ((bb * nh + hh) * p_dim_v + p0) * n_state_v
            var h_run = List[Float32]()
            for i in range(cells):
                h_run.append(sh_p.unsafe_load(sbase + i))
            var h_sealed = h_run.copy()
            var vs = _zeros(q_v * pw)
            var ks = _zeros(q_v * n_state_v)
            for c in range(nc):
                var pbase = (((bb * nc + c) * nh + hh) * p_dim_v + p0) * n_state_v
                for i in range(cells):
                    pass_p.unsafe_store(pbase + i, h_run[i])
                if c == nc - 1:
                    for i in range(cells):
                        h_sealed[i] = h_run[i]
                var c0 = c * q_v
                var real = t_work - c0
                if real > q_v:
                    real = q_v
                var dl = ftz(
                    dacs_p.unsafe_load(((bb * nh + hh) * nc + c) * q_v + (q_v - 1))
                )
                for i in range(q_v * pw):
                    vs[i] = Float32(0.0)
                for i in range(q_v * n_state_v):
                    ks[i] = Float32(0.0)
                for i in range(real):
                    var drev = ftz(
                        dl
                        - ftz(
                            dacs_p.unsafe_load(((bb * nh + hh) * nc + c) * q_v + i)
                        )
                    )
                    var e = ftz(identical_exp(drev))
                    for pl in range(pw):
                        vs[i * pw + pl] = ftz(
                            identical_mul(
                                ftz(
                                    vwork_p.unsafe_load(
                                        ((bb * t_work + c0 + i) * nh + hh)
                                        * p_dim_v
                                        + p0 + pl
                                    )
                                ),
                                e,
                            )
                        )
                    for n in range(n_state_v):
                        ks[i * n_state_v + n] = ksw_p.unsafe_load(
                            ((bb * t_work + c0 + i) * nh + hh) * n_state_v
                            + n
                        )
                var inc = List[Float32]()
                try:
                    inc = gemm_host_rows_right_zero_padded(
                        vs, ks, OP_TN, pw, n_state_v, q_v, real
                    )
                except:
                    # The shapes are the oracle's own, so this cannot
                    # fire; if it ever does, the caller raises after the
                    # region instead of this task swallowing it.
                    err_p.unsafe_store(0, Float32(1.0))
                    return
                var scale_c = ftz(identical_exp(dl))
                for i in range(cells):
                    h_run[i] = ftz(
                        identical_mul_add(
                            scale_c, ftz(h_run[i]), ftz(inc[i])
                        )
                    )
            for i in range(cells):
                hlast_p.unsafe_store(sbase + i, h_run[i])
                sh_p.unsafe_store(sbase + i, h_sealed[i])
    if ptasks <= 1:
        _inter_chunk_rows(0)
    else:
        host_parallelize(_inter_chunk_rows, ptasks)
    if err_flag[0] != Float32(0.0):
        raise Error("mamba3_block_oracle: the inter-chunk product refused its operands")

    host_tick(hton, htk, "m3.inter_chunk")
    # ---- S16 (intra-chunk attention) + S17 (state read-out) + S18 (skip)
    # + S19 (gate); (batch, chunk, head) rows over tasks ---------------------
    st.yintra_out = _zeros(m * nh * p_dim)
    st.ystate_out = _zeros(m * nh * p_dim)
    st.skip_out = _zeros(m * nh * p_dim)
    st.gate_out = _zeros(m * nh * p_dim)
    var yintra_p = _p(st.yintra_out)
    var ystate_p = _p(st.ystate_out)
    var skip_p = _p(st.skip_out)
    var gate_p = _p(st.gate_out)
    var rotqw_p = _p(rotq_work)
    var c_z = dims.col_z()
    var ktasks = host_row_tasks(ch_rows, 4 * q * q * n_state)
    if not chunk_gemms_serial:
        ktasks = 1
    var kchunk = (ch_rows + ktasks - 1) // ktasks
    def _chunk_rows(task: Int) {imm rotqw_p, imm ksw_p, imm vwork_p, imm segl_p, imm pass_p, imm dacs_p, imm qkd_p, imm ip_p, imm dskip_p, imm yintra_p, imm ystate_p, imm skip_p, imm gate_p, imm ch_rows, imm l, imm nh, imm nc, imm dip, imm c_z, imm q0, imm t_work, imm q_v, imm p_dim_v, imm n_state_v, imm kchunk}:
        for r in range(task * kchunk, min((task + 1) * kchunk, ch_rows)):
            var bb = r // (nc * nh)
            var c = (r // nh) % nc
            var hh = r % nh
            var c0 = c * q_v
            var real = t_work - c0
            if real > q_v:
                real = q_v
            var qmat = _zeros(q_v * n_state_v)
            var kmat = _zeros(q_v * n_state_v)
            var vmat = _zeros(q_v * p_dim_v)
            for i in range(real):
                for n in range(n_state_v):
                    qmat[i * n_state_v + n] = rotqw_p.unsafe_load(
                        ((bb * t_work + c0 + i) * nh + hh) * n_state_v
                        + n
                    )
                    kmat[i * n_state_v + n] = ksw_p.unsafe_load(
                        ((bb * t_work + c0 + i) * nh + hh) * n_state_v
                        + n
                    )
                for p in range(p_dim_v):
                    vmat[i * p_dim_v + p] = vwork_p.unsafe_load(
                        ((bb * t_work + c0 + i) * nh + hh) * p_dim_v + p
                    )
            var smat = gemm_host_rows(
                qmat, kmat, OP_NT, real, q_v, n_state_v
            )
            var lbase = (((bb * nc + c) * nh + hh) * q_v) * q_v
            var m_mat = _zeros(q_v * q_v)
            for i in range(real):
                for j in range(i):
                    m_mat[i * q_v + j] = ftz(
                        identical_mul(
                            ftz(smat[i * q_v + j]),
                            ftz(segl_p.unsafe_load(lbase + i * q_v + j)),
                        )
                    )
            var yint = gemm_host_rows(
                m_mat, vmat, OP_NN, real, p_dim_v, q_v
            )
            var h_in = _zeros(p_dim_v * n_state_v)
            var pbase = (((bb * nc + c) * nh + hh) * p_dim_v) * n_state_v
            for i in range(p_dim_v * n_state_v):
                h_in[i] = pass_p.unsafe_load(pbase + i)
            var ch = gemm_host_rows(
                qmat, h_in, OP_NT, real, p_dim_v, n_state_v
            )
            for i in range(real):
                var t = c0 + i
                if t < q0:
                    continue  # buffered rows re-emit no outputs
                var li = t - q0
                var mm = bb * l + li
                var e_i = ftz(
                    identical_exp(
                        ftz(
                            dacs_p.unsafe_load(
                                ((bb * nh + hh) * nc + c) * q_v + i
                            )
                        )
                    )
                )
                for p in range(p_dim_v):
                    var yi = yint[i * p_dim_v + p]
                    var ys = ftz(
                        identical_mul(ftz(ch[i * p_dim_v + p]), e_i)
                    )
                    yintra_p.unsafe_store((mm * nh + hh) * p_dim_v + p, yi)
                    ystate_p.unsafe_store((mm * nh + hh) * p_dim_v + p, ys)
                    var y0 = ftz(ys + yi)
                    var tv = ftz(
                        ftz(dskip_p.unsafe_load(hh))
                        + qkd_p.unsafe_load(mm * nh + hh)
                    )
                    var pv = ftz(
                        identical_mul(
                            tv,
                            ftz(
                                vwork_p.unsafe_load(
                                    ((bb * t_work + t) * nh + hh)
                                    * p_dim_v
                                    + p
                                )
                            ),
                        )
                    )
                    var sk = ftz(y0 + pv)
                    skip_p.unsafe_store((mm * nh + hh) * p_dim_v + p, sk)
                    var zv = ftz(
                        ip_p.unsafe_load(mm * dip + c_z + hh * p_dim_v + p)
                    )
                    gate_p.unsafe_store((mm * nh + hh) * p_dim_v + p, ftz(
                        identical_mul(ftz(sk), ftz(identical_silu(zv)))
                    ))
    if ktasks <= 1:
        _chunk_rows(0)
    else:
        host_parallelize(_chunk_rows, ktasks)

    host_tick(hton, htk, "m3.chunk_attention")
    # ---- reports: k_last (post-bias post-rotation PRE-scale), v_last -----
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
    # chunk's rows ------------------------------------------------------------
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

    host_tick(hton, htk, "m3.reports_buffer")
    # ---- S4: out_proj (mamba3.py:277), gemm v1 OP_NT, k = d_inner. ------
    st.out_proj = gemm_host_rows(st.gate_out, w.w_out, OP_NT, m, dm, di)

    host_tick(hton, htk, "m3.out_proj")
    # ---- S23: residual (block.py:52/:67), mamba2 S22 VERBATIM; cells over
    # tasks, the same statement --------------------------------------------
    st.residual_out = host_f32_uninit(m * dm)
    var res_p = _p(st.residual_out)
    var op_p = _p(st.out_proj)
    var cells = m * dm
    var etasks = host_row_tasks(cells, 3)
    var echunk = (cells + etasks - 1) // etasks
    def _residual_cells(task: Int) {imm xp, imm op_p, imm res_p, imm cells, imm echunk}:
        for i in range(task * echunk, min((task + 1) * echunk, cells)):
            res_p.unsafe_store(i, ftz(ftz(xp.unsafe_load(i)) + op_p.unsafe_load(i)))
    if etasks <= 1:
        _residual_cells(0)
    else:
        host_parallelize(_residual_cells, etasks)
    host_tick(hton, htk, "m3.residual")
    # KEEP-ALIVES. Mojo destroys a local at its last use BY NAME, and the
    # task closures above reach these lists through `_p` pointers only, so
    # without a later use each would be freed while its pointer is still
    # written (measured: the free-list corruption that crashed the first
    # build of this lane). Every local list a pointer was taken of ends here.
    _ = rotq_work^
    _ = rotk_work^
    _ = v_work^
    _ = dt_work^
    _ = sig_work^
    _ = adt_work^
    _ = gamma_work^
    _ = betap_work^
    _ = scale_work^
    _ = kscale_work^
    _ = err_flag^

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
