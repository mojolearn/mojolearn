# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The host Float32 oracle of one Mamba-2 (SSD) block under profile
`mojolearn.identical.mamba2.fp32.v1`, and the Float64 tolerance reference.

NOT A PORT -- the reference libraries ship no oracle; the ALGORITHM below is
theirs, cited seam by seam in `mamba/IDENTICAL_MAMBA2_CONTRACT.md` (commit
e3b46e95). This file is the contract's arithmetic order written on the CPU
through the SAME seam functions the Mamba-1 lane certified
(`identical_mul_add`, `ftz`, `identical_exp`, `identical_div`,
`identical_rsqrt`, `identical_silu`, `identical_softplus`, and DEVIATION
788's `identical_clamp`) plus GEMM v1's `gemm_oracle`, so the device card of
`mamba/impl/mamba_ssm/modules/mamba2.mojo` / `ssd_minimal.mojo` is diffed
against it bitwise. The device kernels are an INDEPENDENT transcription of
the same order; nothing below is imported by them except the seam functions
themselves (one arithmetic, two spellings of the loops around it -- the
Mamba-1 oracle's rule, unchanged).

INHERITED MACHINERY, NOT RESPELLED. The contract's inheritance column is
honored by construction: S1-S3 and S21's fold/rstd/product spelling is the
Mamba-1 oracle's RMSNorm machinery character for character; S6 is its
bias-SEEDED conv chain (DEVIATION 721 adopted verbatim); S5 its
`A = -exp(A_log)`; S7/S8 `identical_silu`; S9's softplus is DEVIATION 745's
function; every projection and every intra-chunk contraction is a
`gemm.fp32.v1` cell through `gemm_oracle` (DEVIATION 784). A second spelling
of an inherited seam is a defect, so where a loop below matches
`mamba_oracle.mojo` line for line, that is the point.

DECODE IS PREFILL RESUMPTION (DEVIATION 786, contract section 5). ONE
function serves both paths: `mamba2_block_oracle` takes the three-piece
state (conv window, boundary h, intra-chunk buffer), rebuilds the OPEN
chunk's working sequence (buffered rows ++ new rows), and runs the chunked
SSD over it. A prefill is the same call with a fresh zero state. The padded
folds a decode step runs over the open chunk are bitwise the folds a
prefill ending at that token runs -- that is the construction gate D then
verifies. The upstream per-token step recurrence
(`selective_state_update_ref`:277-282) is NOT here; it lives in the impl
file as the required-RED arm STEP_UPSTREAM_RECURRENCE.

DEVIATION 790 (NEW, this file, the impl and the check cite it) -- THE
WORKING-SEQUENCE STAGE SHAPES, and gate (d)'s comparability clause.
Contract section 7 gives the chunk-shaped stages (`dacs.out`, `seg.L`,
`cb.G`, `decay.states`, `cstate.out`, `pass.states`) shapes in
`C = ceil(L/Q)` chunks of THE CALL. On a RESUMED call the SSD core runs
over the working sequence (q0 buffered rows ++ l new rows), so those
stages cover the WORKING chunks -- on a fresh prefill (q0 = 0) the two
readings coincide and section 7 is satisfied verbatim; on a decode step
they cover exactly the open chunk. Consequence for gate (d): the
token-shaped stages compare bitwise PER TOKEN (each decode row equals the
prefill's row for that token); the chunk-shaped stages compare bitwise
only where both sides have the same fill level -- `pass.states` (the state
ENTERING a chunk) at any completed boundary, everything else at the FINAL
decoded token, where the working chunk's real-row count equals the
prefill's. The check spells exactly that, and says so in its banner. No
arithmetic moves under this deviation; it is a stage-addressing clause.

Token-shaped SSD stages (`xd.out`, `ydiag.out`, `yoff.out`, `scan.y`,
`skip.out`, ...) are recorded for the NEW tokens only (`M = B * l` rows),
sliced from the working sequence -- a copy, not a seam.

The Float64 reference at the bottom is the discrete SSD OPERATOR (the
per-token recurrence ssd_minimal_discrete factorizes, exact algebra) in
double precision. It is a TOLERANCE instrument, never a bitwise one.
"""

from checks.numerics import (
    ftz,
    identical_clamp,
    identical_div,
    identical_exp,
    identical_mul_add,
    identical_rsqrt,
    identical_silu,
    identical_softplus,
)
from gemm.checks.gemm_oracle import OP_NN, OP_NT, OP_TN, gemm_oracle
from mamba.checks.mamba_oracle import refuse_nonfinite
from mamba.checks.mamba2_fixture import (
    M2_CHUNK_SIZE,
    M2_D_CONV,
    M2_D_STATE,
    M2_HEADDIM,
    M2_RMS_EPS,
    Mamba2Dims,
    Mamba2Weights,
)


def pinned_mul(a: Float32, b: Float32) -> Float32:
    """DEVIATION 720's construction, the Mamba-1 oracle's copy: a MULTIPLY
    no codegen may contract into a neighboring add, spelled
    `identical_mul_add(a, b, -0.0)` (bit-equal to the correctly rounded
    product at every input including both zero signs; the `+0.0` addend
    would launder a `-0.0` product, gemm F6a). Used at every seam the
    contract marks PRODUCT: S3's two products, S10's two discretizations,
    S13, S15's `B ⊙ decay`, S18's decay scale, S20's `x * D`, S21's gate
    and products."""
    return identical_mul_add(a, b, Float32(-0.0))


def m2_refuse_bad_inputs(
    w: Mamba2Weights, x: List[Float32], state: Mamba2State
) raises:
    """Contract section 6: refusal BY NAME and BY BITS before any recorded
    stage, `mamba_oracle.mojo::refuse_nonfinite` (row 39) reused. The
    segsum's `-inf` never exists in a buffer (DEVIATION 782, structural
    zeros), so nothing here needs an exemption; the dt_limit SCALARS are
    runtime inputs, not buffers, and `+inf` is legal there by contract."""
    refuse_nonfinite("x", x)
    refuse_nonfinite("norm.weight", w.norm_w)
    refuse_nonfinite("in_proj.weight", w.w_in)
    refuse_nonfinite("conv1d.weight", w.conv_w)
    refuse_nonfinite("conv1d.bias", w.conv_b)
    refuse_nonfinite("dt_bias", w.dt_bias)
    refuse_nonfinite("A_log", w.a_log)
    refuse_nonfinite("D", w.d_skip)
    refuse_nonfinite("norm_gated.weight", w.gnorm_w)
    refuse_nonfinite("out_proj.weight", w.w_out)
    refuse_nonfinite("state.conv_win", state.conv_win)
    refuse_nonfinite("state.h", state.h)
    refuse_nonfinite("state.buf_xbc", state.buf_xbc)
    refuse_nonfinite("state.buf_dtraw", state.buf_dtraw)


struct Mamba2State(Copyable, Movable):
    """Contract section 5's THREE-piece recurrent state:

      1. `conv_win` [B, conv_dim, 4]: the last d_conv PRE-conv xBC inputs
         per channel, oldest first, zeros before the first token
         (`allocate_inference_cache`:348-350);
      2. `h` [B, H, P, N]: the S17 running value as of the last COMPLETED
         chunk (zeros, or `initial_states`, before the first);
      3. the intra-chunk buffer: `buf_len` rows (< Q) of post-conv/
         post-SiLU xBC (`buf_xbc` [B, Q, conv_dim]) and RAW dt
         (`buf_dtraw` [B, Q, H]) for the open chunk, enough to re-derive
         every S10-S19 quantity (S9 is re-derived too -- a pure function
         of the same bits).

    One `buf_len` for the whole batch: every sequence in a launch advances
    together (varlen is deferred by name, contract section 5)."""

    var b: Int
    var dims: Mamba2Dims
    var buf_len: Int
    var conv_win: List[Float32]
    var h: List[Float32]
    var buf_xbc: List[Float32]
    var buf_dtraw: List[Float32]

    def __init__(out self, b: Int, dims: Mamba2Dims):
        self.b = b
        self.dims = dims.copy()
        self.buf_len = 0
        self.conv_win = List[Float32]()
        self.h = List[Float32]()
        self.buf_xbc = List[Float32]()
        self.buf_dtraw = List[Float32]()
        var cd = dims.conv_dim()
        for _ in range(b * cd * M2_D_CONV):
            self.conv_win.append(0.0)
        for _ in range(b * dims.nheads * M2_HEADDIM * M2_D_STATE):
            self.h.append(0.0)
        for _ in range(b * M2_CHUNK_SIZE * cd):
            self.buf_xbc.append(0.0)
        for _ in range(b * M2_CHUNK_SIZE * dims.nheads):
            self.buf_dtraw.append(0.0)

    def set_initial_states(mut self, init: List[Float32]) raises:
        """`initial_states` in (ssd_minimal.py:64-66, prepended as chunk
        -1's state): legal only on a FRESH state -- handing a mid-chunk
        state a new chunk -1 has no upstream meaning."""
        if self.buf_len != 0:
            raise Error(
                "Mamba2State.set_initial_states: the state is mid-chunk"
                " (buf_len = "
                + String(self.buf_len)
                + "); initial_states only has upstream meaning on a fresh"
                " state (ssd_minimal.py:64-66)"
            )
        if len(init) != len(self.h):
            raise Error(
                "Mamba2State.set_initial_states: got "
                + String(len(init))
                + " values for an h of "
                + String(len(self.h))
            )
        for i in range(len(init)):
            self.h[i] = init[i]


struct Mamba2Stages(Movable):
    """Every recorded stage of one block call, contract section 7's card
    order. Token stages are token-major over the call's NEW tokens
    (`M = B * l` rows); chunk stages cover the WORKING chunks (DEVIATION
    790: `n_chunks = ceil((q0 + l) / Q)`, which is `ceil(L/Q)` on a fresh
    prefill). `q0_at_entry` and `t_work` are bookkeeping for the gates,
    not stages."""

    var q0_at_entry: Int
    var t_work: Int
    var n_chunks: Int
    var norm_sumsq: List[Float32]  # [M]                       S1
    var norm_out: List[Float32]  # [M, d_model]                S2-S3
    var in_proj: List[Float32]  # [M, d_in_proj]               S4
    var a_out: List[Float32]  # [H]                            S5
    var conv_out: List[Float32]  # [M, CD]                     S6
    var silu_out: List[Float32]  # [M, CD]                     S7
    var conv_win: List[Float32]  # [B, CD, 4] window AFTER     copies
    var dt_out: List[Float32]  # [M, H]                        S9
    var xd_out: List[Float32]  # [M, H, P]                     S10
    var dacs_out: List[Float32]  # [B, H, C, Q]                S11
    var seg_l: List[Float32]  # [B, C, H, Q, Q]                S12 exp(segsum)
    var cb_g: List[Float32]  # [B, C, G, Q, Q]                 S12 C.B
    var ydiag_out: List[Float32]  # [M, H, P]                  S13-S14
    var decay_states: List[Float32]  # [B, H, C, Q]            S15 exp
    var cstate_out: List[Float32]  # [B, C, H, P, N]           S15-S16
    var pass_states: List[Float32]  # [B, C, H, P, N] entering S17
    var yoff_out: List[Float32]  # [M, H, P]                   S18
    var scan_y: List[Float32]  # [M, H, P]                     S19
    var skip_out: List[Float32]  # [M, H, P]                   S20
    var gnorm_gate: List[Float32]  # [M, d_ssm]                S21 gate
    var gnorm_sumsq: List[Float32]  # [M]                      S21 fold
    var gnorm_out: List[Float32]  # [M, d_ssm]                 S21 out
    var out_proj: List[Float32]  # [M, d_model]                S4
    var residual_out: List[Float32]  # [M, d_model]            S22
    var h_last: List[Float32]  # [B, H, P, N]  section 5's REPORT stage
    var state_h_after: List[Float32]  # [B, H, P, N] the RESUMPTION h (copy)

    def __init__(out self):
        self.q0_at_entry = 0
        self.t_work = 0
        self.n_chunks = 0
        self.norm_sumsq = List[Float32]()
        self.norm_out = List[Float32]()
        self.in_proj = List[Float32]()
        self.a_out = List[Float32]()
        self.conv_out = List[Float32]()
        self.silu_out = List[Float32]()
        self.conv_win = List[Float32]()
        self.dt_out = List[Float32]()
        self.xd_out = List[Float32]()
        self.dacs_out = List[Float32]()
        self.seg_l = List[Float32]()
        self.cb_g = List[Float32]()
        self.ydiag_out = List[Float32]()
        self.decay_states = List[Float32]()
        self.cstate_out = List[Float32]()
        self.pass_states = List[Float32]()
        self.yoff_out = List[Float32]()
        self.scan_y = List[Float32]()
        self.skip_out = List[Float32]()
        self.gnorm_gate = List[Float32]()
        self.gnorm_sumsq = List[Float32]()
        self.gnorm_out = List[Float32]()
        self.out_proj = List[Float32]()
        self.residual_out = List[Float32]()
        self.h_last = List[Float32]()
        self.state_h_after = List[Float32]()


def _zeros(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(0.0)
    return out^


# ===========================================================================
# The SSD core, seams S10-S19 + h_last, on the WORKING sequence.
# Normative reference: ssd_minimal.py::ssd_minimal_discrete (:34-78) composed
# discretize-first (:94-103); orders pinned per contract section 4;
# DEVIATIONS 782 (segsum schedule), 783 (Q part of the arithmetic), 784
# (contractions through gemm v1), 785 (serial state pass), 789
# (discretize-first pairing).
# ===========================================================================


def ssd_core_oracle(
    xbc_work: List[Float32],  # [B, T, CD] post-conv/post-SiLU
    dt_work: List[Float32],  # [B, T, H]  post-S9 (biased, softplussed, clamped)
    a_out: List[Float32],  # [H]        S5's -exp(A_log)
    b: Int,
    t_work: Int,
    dims: Mamba2Dims,
    mut h_boundary: List[Float32],  # [B, H, P, N] in: chunk -1 / last boundary;
    #                                 out: the last COMPLETED boundary
    mut st: Mamba2Stages,
) raises:
    """Fills the SSD stages of `st` (xd through scan.y, h_last) and advances
    `h_boundary` through every COMPLETED working chunk. `st.xd_out` etc. are
    filled for ALL working rows here; the block caller slices the new-token
    rows into the card (DEVIATION 790).

    Only the LAST working chunk can be partial; every reduction below runs
    over the full padded Q (contract section 3's padding clause: a padded
    position contributes an exactly-zero product and moves no nonzero bit;
    padded dacs positions COPY the last real value, section 7)."""
    comptime q = M2_CHUNK_SIZE
    comptime n_state = M2_D_STATE
    comptime p_dim = M2_HEADDIM
    var di = dims.d_inner
    var cd = dims.conv_dim()
    var nh = dims.nheads
    var nc = (t_work + q - 1) // q
    if nc < 1:
        nc = 1
    st.n_chunks = nc

    # Stage buffers at working shapes.
    st.xd_out = _zeros(b * t_work * nh * p_dim)
    st.ydiag_out = _zeros(b * t_work * nh * p_dim)
    st.yoff_out = _zeros(b * t_work * nh * p_dim)
    st.scan_y = _zeros(b * t_work * nh * p_dim)
    st.dacs_out = _zeros(b * nh * nc * q)
    st.seg_l = _zeros(b * nc * nh * q * q)
    st.cb_g = _zeros(b * nc * 1 * q * q)
    st.decay_states = _zeros(b * nh * nc * q)
    st.cstate_out = _zeros(b * nc * nh * p_dim * n_state)
    st.pass_states = _zeros(b * nc * nh * p_dim * n_state)
    st.h_last = _zeros(b * nh * p_dim * n_state)

    for bb in range(b):
        # h entering the first working chunk: the carried boundary state
        # (initial_states or zeros on a fresh call) -- ssd_minimal:64-66.
        var h_run = List[Float32]()
        for i in range(nh * p_dim * n_state):
            h_run.append(h_boundary[bb * nh * p_dim * n_state + i])
        var h_completed = h_run.copy()
        var any_completed = False

        for c in range(nc):
            var c0 = c * q  # first working row of this chunk
            var real = t_work - c0
            if real > q:
                real = q

            # ---- S10, DISCRETIZE FIRST (DEVIATION 789): X_d = x * dt,
            #      dA = dt * A. dt binds to x and to A, never to B.
            #      Padded rows stay exactly +0.0 in every buffer.
            var xd = _zeros(nh * q * p_dim)  # [h][i][p] per chunk
            var da = _zeros(nh * q)  # [h][i]
            for i in range(real):
                var t = c0 + i
                for hh in range(nh):
                    var dtv = ftz(dt_work[(bb * t_work + t) * nh + hh])
                    da[hh * q + i] = ftz(
                        pinned_mul(dtv, ftz(a_out[hh]))
                    )
                    for p in range(p_dim):
                        var xv = ftz(
                            xbc_work[(bb * t_work + t) * cd + hh * p_dim + p]
                        )
                        var v = ftz(pinned_mul(xv, dtv))
                        xd[(hh * q + i) * p_dim + p] = v
                        st.xd_out[
                            ((bb * t_work + t) * nh + hh) * p_dim + p
                        ] = v

            # ---- S11: per-chunk cumsum, SERIAL ASCENDING; padded
            #      positions COPY the last real value (section 7). The
            #      chunk boundary is a hard reset (no cross-chunk carry).
            for hh in range(nh):
                var run = Float32(0.0)
                for i in range(q):
                    if i < real:
                        if i == 0:
                            run = ftz(da[hh * q])
                        else:
                            run = ftz(run + da[hh * q + i])
                    # else: copy of the last real value, not an add
                    st.dacs_out[((bb * nh + hh) * nc + c) * q + i] = run

            # ---- S12: L = exp(segsum), structural zeros above the
            #      diagonal (DEVIATION 782); and G = C.B, a gemm v1 cell
            #      over n (k = 128, one serial leaf).
            var bmat = _zeros(q * n_state)  # [i][n], padded rows +0.0
            var cmat = _zeros(q * n_state)
            for i in range(real):
                var t = c0 + i
                for n in range(n_state):
                    bmat[i * n_state + n] = xbc_work[
                        (bb * t_work + t) * cd + di + n
                    ]
                    cmat[i * n_state + n] = xbc_work[
                        (bb * t_work + t) * cd + di + n_state + n
                    ]
            for hh in range(nh):
                # seg[i][j] for j < i rebuilt column-serial:
                # seg[i][j] = ftz(seg[i-1][j] + dA[i]), seg[j][j] = +0.0;
                # a padded row's dA never exists, so its entry is a copy.
                var lbase = (((bb * nc + c) * nh + hh) * q) * q
                for j in range(q):
                    var acc = Float32(0.0)
                    st.seg_l[lbase + j * q + j] = ftz(identical_exp(acc))
                    for i in range(j + 1, q):
                        if i < real:
                            acc = ftz(acc + da[hh * q + i])
                        st.seg_l[lbase + i * q + j] = ftz(identical_exp(acc))
                    # above the diagonal: +0.0, already the zero fill --
                    # STRUCTURAL, never a computed exp(-inf).
            # G, per group (G = 1): [Q, Q] = C[Q, N] . B[Q, N]^T.
            var g_mat = gemm_oracle(cmat, bmat, OP_NT, q, q, n_state)
            var gbase = ((bb * nc + c) * 1) * q * q
            for i in range(q * q):
                st.cb_g[gbase + i] = g_mat[i]

            # ---- S13 (M = G ⊙ L, PRODUCT) + S14 (Y_diag = M . X_d, gemm
            #      cell at k = Q = 256: two leaves, one fold level).
            for hh in range(nh):
                var lbase = (((bb * nc + c) * nh + hh) * q) * q
                var m_mat = _zeros(q * q)
                for i in range(q):
                    for j in range(i + 1):
                        m_mat[i * q + j] = ftz(
                            pinned_mul(
                                ftz(g_mat[i * q + j]),
                                ftz(st.seg_l[lbase + i * q + j]),
                            )
                        )
                    # j > i stays +0.0: structurally zero M entries enter
                    # the S14 fold as exact zeros.
                var xd_chunk = _zeros(q * p_dim)
                for i in range(q):
                    for p in range(p_dim):
                        xd_chunk[i * p_dim + p] = xd[(hh * q + i) * p_dim + p]
                var ydiag = gemm_oracle(m_mat, xd_chunk, OP_NN, q, p_dim, q)
                for i in range(real):
                    var t = c0 + i
                    for p in range(p_dim):
                        st.ydiag_out[
                            ((bb * t_work + t) * nh + hh) * p_dim + p
                        ] = ydiag[i * p_dim + p]

                # ---- S15: decay = exp(dA_cs_last - dA_cs) (one
                #      subtraction), B_decay = B ⊙ decay.
                var dacs_last = st.dacs_out[
                    ((bb * nh + hh) * nc + c) * q + (q - 1)
                ]
                var bd = _zeros(q * n_state)
                for i in range(q):
                    var dacs_i = st.dacs_out[((bb * nh + hh) * nc + c) * q + i]
                    var d = ftz(dacs_last - dacs_i)
                    var dec = ftz(identical_exp(d))
                    st.decay_states[((bb * nh + hh) * nc + c) * q + i] = dec
                    if i < real:
                        for n in range(n_state):
                            bd[i * n_state + n] = ftz(
                                pinned_mul(ftz(bmat[i * n_state + n]), dec)
                            )
                    # padded rows: B is +0.0, so B_decay stays +0.0 and the
                    # S16 fold sees exact zeros.

                # ---- S16: chunk_states = B_decay^T . X_d over chunk
                #      positions (k = Q = 256), output [P, N] per (b,c,h).
                var cstate = gemm_oracle(xd_chunk, bd, OP_TN, p_dim, n_state, q)
                var cbase = (((bb * nc + c) * nh + hh) * p_dim) * n_state
                for i in range(p_dim * n_state):
                    st.cstate_out[cbase + i] = cstate[i]

                # ---- S17: SERIAL inter-chunk pass (DEVIATION 785), the
                #      fused fma per (b, h, p, n); pass.states records the
                #      state ENTERING this chunk.
                var pbase = cbase  # same [B, C, H, P, N] layout
                var scale = ftz(identical_exp(ftz(dacs_last)))
                var h_prev = _zeros(p_dim * n_state)
                for i in range(p_dim * n_state):
                    h_prev[i] = h_run[(hh * p_dim) * n_state + i]
                    st.pass_states[pbase + i] = h_prev[i]
                for i in range(p_dim * n_state):
                    h_run[(hh * p_dim) * n_state + i] = ftz(
                        identical_mul_add(
                            scale, ftz(h_prev[i]), ftz(cstate[i])
                        )
                    )

                # ---- S18: Y_off = (C . h_prev) ⊙ exp(dA_cs) -- contract
                #      over n FIRST (gemm cell, k = 128), scale AFTER.
                var ch = gemm_oracle(cmat, h_prev, OP_NT, q, p_dim, n_state)
                for i in range(real):
                    var t = c0 + i
                    var dacs_i = st.dacs_out[((bb * nh + hh) * nc + c) * q + i]
                    var sc = ftz(identical_exp(ftz(dacs_i)))
                    for p in range(p_dim):
                        var yo = ftz(pinned_mul(ftz(ch[i * p_dim + p]), sc))
                        st.yoff_out[
                            ((bb * t_work + t) * nh + hh) * p_dim + p
                        ] = yo
                        # ---- S19: Y = Y_diag + Y_off, one add.
                        st.scan_y[
                            ((bb * t_work + t) * nh + hh) * p_dim + p
                        ] = ftz(
                            st.ydiag_out[
                                ((bb * t_work + t) * nh + hh) * p_dim + p
                            ]
                            + yo
                        )

            if real == q:
                # This chunk COMPLETED: its exit state is a resumption
                # boundary (contract section 5's piece 2).
                any_completed = True
                for i in range(nh * p_dim * n_state):
                    h_completed[i] = h_run[i]

        # h_last: the S17 value after the FINAL (padded) chunk -- the
        # REPORT stage (upstream's final_state, ssd_minimal.py:69), not the
        # resumption state.
        for i in range(nh * p_dim * n_state):
            st.h_last[bb * nh * p_dim * n_state + i] = h_run[i]
        if any_completed:
            for i in range(nh * p_dim * n_state):
                h_boundary[bb * nh * p_dim * n_state + i] = h_completed[i]
        # else: the boundary stays what it was (no chunk completed).


# ===========================================================================
# The block, seams S1-S9 and S20-S22 composed around the core.
# Order: Mamba2Block.forward (HF :617-631) around Mamba2.forward's
# non-mem-eff arm (mamba2.py:209-276), contract section 2.
# ===========================================================================


def mamba2_block_oracle(
    w: Mamba2Weights,
    x: List[Float32],  # [B, l, d_model] token-major -- the NEW tokens
    b: Int,
    l: Int,
    dt_lo: Float32,
    dt_hi: Float32,
    mut state: Mamba2State,
) raises -> Mamba2Stages:
    """One block call: prefill at a fresh zero `Mamba2State`, decode at
    `l == 1` (or any l) carrying it -- ONE spelling for both (DEVIATION
    786). Returns the section 7 card; mutates the three-piece state."""
    m2_refuse_bad_inputs(w, x, state)
    var dims = w.dims.copy()
    var dm = dims.d_model
    var di = dims.d_inner
    var cd = dims.conv_dim()
    var dip = dims.d_in_proj()
    var nh = dims.nheads
    comptime p_dim = M2_HEADDIM
    comptime q = M2_CHUNK_SIZE
    var m = b * l
    if len(x) != m * dm:
        raise Error(
            String("mamba2_block_oracle: ")
            + String(len(x))
            + " input values for B = "
            + String(b)
            + ", l = "
            + String(l)
            + ", d_model = "
            + String(dm)
        )
    # Any working length is legal: the sequence may span several chunks in
    # one call, and only the LAST working chunk can be partial.
    var st = Mamba2Stages()
    st.q0_at_entry = state.buf_len

    # ---- block RMSNorm (HF Mamba2RMSNorm :591-605), S1-S3: the Mamba-1
    #      S1-S4 machinery verbatim (inheritance column).
    for t in range(m):
        var acc = Float32(0.0)
        for j in range(dm):
            var xj = ftz(x[t * dm + j])
            acc = ftz(identical_mul_add(xj, xj, acc))
        st.norm_sumsq.append(acc)
        var mean = ftz(identical_div(acc, Float32(dm)))
        var rstd = ftz(identical_rsqrt(ftz(mean + M2_RMS_EPS)))
        for j in range(dm):
            var inner = ftz(pinned_mul(ftz(x[t * dm + j]), rstd))
            st.norm_out.append(ftz(pinned_mul(ftz(w.norm_w[j]), inner)))

    # ---- in_proj (mamba2.py:211; Linear, bias=False), S4: gemm v1 OP_NT,
    #      k = d_model. Columns z | xBC | dt_raw (:211-215 order).
    st.in_proj = gemm_oracle(st.norm_out, w.w_in, OP_NT, m, dip, dm)

    # ---- A = -exp(A_log) (mamba2.py:182), S5. PER HEAD.
    for hh in range(nh):
        st.a_out.append(-ftz(identical_exp(ftz(w.a_log[hh]))))

    # ---- causal depthwise conv over xBC + SiLU (mamba2.py:230-242),
    #      S6/S7: the Mamba-1 conv chain (bias-SEEDED, taps ascending,
    #      DEVIATION 721 adopted verbatim -- mamba1 S13's spelling), over
    #      conv_dim channels; window carries PRE-conv xBC values.
    st.conv_out = _zeros(m * cd)
    st.silu_out = _zeros(m * cd)
    var new_win = List[Float32]()
    for bb in range(b):
        for d in range(cd):
            for li in range(l):
                var acc = ftz(w.conv_b[d])
                for k in range(M2_D_CONV):
                    var p = li - (M2_D_CONV - 1) + k
                    var xv: Float32
                    if p >= 0:
                        xv = st.in_proj[(bb * l + p) * dip + di + d]
                    else:
                        xv = state.conv_win[
                            (bb * cd + d) * M2_D_CONV + (M2_D_CONV + p)
                        ]
                    acc = ftz(
                        identical_mul_add(
                            ftz(w.conv_w[d * M2_D_CONV + k]), ftz(xv), acc
                        )
                    )
                st.conv_out[(bb * l + li) * cd + d] = acc
                st.silu_out[(bb * l + li) * cd + d] = ftz(identical_silu(acc))
            for j in range(M2_D_CONV):
                var p = l - M2_D_CONV + j
                if p >= 0:
                    new_win.append(st.in_proj[(bb * l + p) * dip + di + d])
                else:
                    new_win.append(
                        state.conv_win[
                            (bb * cd + d) * M2_D_CONV + (M2_D_CONV + p)
                        ]
                    )
    state.conv_win = new_win.copy()
    st.conv_win = new_win^

    # ---- S9 for the NEW rows: dt = identical_clamp(softplus(dt_raw +
    #      bias), lo, hi) -- bias, softplus, clamp IN THAT ORDER
    #      (_chunk_cumsum_fwd_kernel:73-81; DEVIATION 788's primitive).
    #      Present and pinned even at the inert default.
    for t in range(m):
        for hh in range(nh):
            var raw = st.in_proj[t * dip + di + cd + hh]
            var biased = ftz(ftz(raw) + ftz(w.dt_bias[hh]))
            var sp = ftz(identical_softplus(biased))
            st.dt_out.append(ftz(identical_clamp(sp, dt_lo, dt_hi)))

    # ---- assemble the WORKING sequence (DEVIATION 786/790): buffered
    #      rows ++ new rows. Copies, not seams. S9 is RE-DERIVED for the
    #      buffered rows from their raw dt -- a pure function of the same
    #      bits, so the re-derivation is bitwise the original.
    var q0 = state.buf_len
    var t_work = q0 + l
    st.t_work = t_work
    var xbc_work = _zeros(b * t_work * cd)
    var dt_work = _zeros(b * t_work * nh)
    for bb in range(b):
        for t in range(q0):
            for d in range(cd):
                xbc_work[(bb * t_work + t) * cd + d] = state.buf_xbc[
                    (bb * q + t) * cd + d
                ]
            for hh in range(nh):
                var raw = state.buf_dtraw[(bb * q + t) * nh + hh]
                var biased = ftz(ftz(raw) + ftz(w.dt_bias[hh]))
                var sp = ftz(identical_softplus(biased))
                dt_work[(bb * t_work + t) * nh + hh] = ftz(
                    identical_clamp(sp, dt_lo, dt_hi)
                )
        for li in range(l):
            var t = q0 + li
            for d in range(cd):
                xbc_work[(bb * t_work + t) * cd + d] = st.silu_out[
                    (bb * l + li) * cd + d
                ]
            for hh in range(nh):
                dt_work[(bb * t_work + t) * nh + hh] = st.dt_out[
                    (bb * l + li) * nh + hh
                ]

    # ---- the SSD core, S10-S19 + h_last, boundary h advanced through
    #      completed chunks. `st.a_out` may not ride along in the same
    #      argument list as the mutable `st` (the Sep 1 aliasing rule: one
    #      value may not be passed immutably and mutably in one call), so
    #      the core reads its OWN copy of A -- a copy of [H] values is not
    #      a seam and moves no bits.
    var a_vals = st.a_out.copy()
    ssd_core_oracle(xbc_work, dt_work, a_vals, b, t_work, dims, state.h, st)

    # ---- slice the token-shaped SSD stages to the NEW tokens (DEVIATION
    #      790: a copy, not a seam). Working row of new token (bb, li) is
    #      q0 + li.
    var xd_new = _zeros(m * nh * p_dim)
    var ydiag_new = _zeros(m * nh * p_dim)
    var yoff_new = _zeros(m * nh * p_dim)
    var y_new = _zeros(m * nh * p_dim)
    for bb in range(b):
        for li in range(l):
            var t = q0 + li
            for i in range(nh * p_dim):
                xd_new[(bb * l + li) * nh * p_dim + i] = st.xd_out[
                    (bb * t_work + t) * nh * p_dim + i
                ]
                ydiag_new[(bb * l + li) * nh * p_dim + i] = st.ydiag_out[
                    (bb * t_work + t) * nh * p_dim + i
                ]
                yoff_new[(bb * l + li) * nh * p_dim + i] = st.yoff_out[
                    (bb * t_work + t) * nh * p_dim + i
                ]
                y_new[(bb * l + li) * nh * p_dim + i] = st.scan_y[
                    (bb * t_work + t) * nh * p_dim + i
                ]
    st.xd_out = xd_new^
    st.ydiag_out = ydiag_new^
    st.yoff_out = yoff_new^
    st.scan_y = y_new^

    # ---- update the intra-chunk buffer: the last (t_work mod Q) working
    #      rows, or empty when the final chunk completed. Copies.
    var r = t_work - (t_work // q) * q
    for bb in range(b):
        for t in range(r):
            var src = t_work - r + t
            for d in range(cd):
                state.buf_xbc[(bb * q + t) * cd + d] = xbc_work[
                    (bb * t_work + src) * cd + d
                ]
        # raw dt rows: new rows carry in_proj's dt_raw; still-buffered old
        # rows keep their stored raw values.
        for t in range(r):
            var src = t_work - r + t
            for hh in range(nh):
                var raw: Float32
                if src < q0:
                    raw = state.buf_dtraw[(bb * q + src) * nh + hh]
                else:
                    var li = src - q0
                    raw = st.in_proj[(bb * l + li) * dip + di + cd + hh]
                state.buf_dtraw[(bb * q + t) * nh + hh] = raw
    state.buf_len = r

    # ---- S20: out = Y + x * D[h], D LAST, x UNDISCRETIZED post-conv
    #      (HF :284-285, :338-339; mamba1 S11's decision).
    for t in range(m):
        for hh in range(nh):
            for p in range(p_dim):
                var xv = ftz(st.silu_out[t * cd + hh * p_dim + p])
                var prod = ftz(pinned_mul(xv, ftz(w.d_skip[hh])))
                st.skip_out.append(
                    ftz(st.scan_y[(t * nh + hh) * p_dim + p] + prod)
                )

    # ---- S21: gated RMSNorm, gate FIRST (norm_before_gate=False,
    #      rms_norm_ref:26-27; DEVIATION 787), then the S1-S3 machinery
    #      over d_ssm = d_inner (group_size = whole row at G = 1).
    for t in range(m):
        for j in range(di):
            var z = ftz(st.in_proj[t * dip + j])
            st.gnorm_gate.append(
                ftz(
                    pinned_mul(
                        ftz(st.skip_out[t * di + j]), ftz(identical_silu(z))
                    )
                )
            )
    for t in range(m):
        var acc = Float32(0.0)
        for j in range(di):
            var g = ftz(st.gnorm_gate[t * di + j])
            acc = ftz(identical_mul_add(g, g, acc))
        st.gnorm_sumsq.append(acc)
        var mean = ftz(identical_div(acc, Float32(di)))
        var rstd = ftz(identical_rsqrt(ftz(mean + M2_RMS_EPS)))
        for j in range(di):
            var inner = ftz(pinned_mul(ftz(st.gnorm_gate[t * di + j]), rstd))
            st.gnorm_out.append(ftz(pinned_mul(ftz(w.gnorm_w[j]), inner)))

    # ---- out_proj (mamba2.py:275), S4: gemm v1 OP_NT, k = d_inner.
    st.out_proj = gemm_oracle(st.gnorm_out, w.w_out, OP_NT, m, dm, di)

    # ---- S22: residual (HF :630).
    for i in range(m * dm):
        st.residual_out.append(ftz(ftz(x[i]) + st.out_proj[i]))

    # A readable copy of the resumption h for the gates.
    for i in range(len(state.h)):
        st.state_h_after.append(state.h[i])
    return st^


# ===========================================================================
# The Float64 tolerance reference: the discrete SSD OPERATOR as its exact
# per-token recurrence (h_t = exp(dt_t A) h_{t-1} + B_t (x_t dt_t),
# y_t = C_t . h_t -- the algebra ssd_minimal_discrete factorizes), plain
# double spellings. A TOLERANCE instrument; bitwise claims never touch it.
# ===========================================================================


struct Mamba2Ref64(Movable):
    var skip_out: List[Float64]  # [M, H, P]  y + x*D
    var residual_out: List[Float64]  # [M, d_model]

    def __init__(out self):
        self.skip_out = List[Float64]()
        self.residual_out = List[Float64]()


def mamba2_block_ref64(
    w: Mamba2Weights,
    x: List[Float32],
    b: Int,
    l: Int,
    dt_lo: Float32,
    dt_hi: Float32,
    init_h: List[Float32],  # [B, H, P, N], zeros for none
) raises -> Mamba2Ref64:
    from std.math import exp, log, sqrt

    var dims = w.dims.copy()
    var dm = dims.d_model
    var di = dims.d_inner
    var cd = dims.conv_dim()
    var dip = dims.d_in_proj()
    var nh = dims.nheads
    comptime p_dim = M2_HEADDIM
    comptime n_state = M2_D_STATE
    var m = b * l
    var out = Mamba2Ref64()

    var norm = List[Float64]()
    for t in range(m):
        var acc = Float64(0.0)
        for j in range(dm):
            acc += Float64(x[t * dm + j]) * Float64(x[t * dm + j])
        var rstd = 1.0 / sqrt(acc / Float64(dm) + Float64(M2_RMS_EPS))
        for j in range(dm):
            norm.append(Float64(w.norm_w[j]) * (Float64(x[t * dm + j]) * rstd))

    var proj = List[Float64]()
    for t in range(m):
        for c in range(dip):
            var acc = Float64(0.0)
            for j in range(dm):
                acc += norm[t * dm + j] * Float64(w.w_in[c * dm + j])
            proj.append(acc)

    # conv + silu over xBC, zero pre-sequence window (fresh call only).
    var xbc = List[Float64]()
    for _ in range(m * cd):
        xbc.append(0.0)
    for bb in range(b):
        for d in range(cd):
            for li in range(l):
                var acc = Float64(w.conv_b[d])
                for k in range(M2_D_CONV):
                    var p = li - (M2_D_CONV - 1) + k
                    var xv = Float64(0.0)
                    if p >= 0:
                        xv = proj[(bb * l + p) * dip + di + d]
                    acc += Float64(w.conv_w[d * M2_D_CONV + k]) * xv
                xbc[(bb * l + li) * cd + d] = acc / (1.0 + exp(-acc))

    var dt = List[Float64]()
    for t in range(m):
        for hh in range(nh):
            var v = proj[t * dip + di + cd + hh] + Float64(w.dt_bias[hh])
            if v <= 20.0:
                v = log(1.0 + exp(v))
            if v < Float64(dt_lo):
                v = Float64(dt_lo)
            if v > Float64(dt_hi):
                v = Float64(dt_hi)
            dt.append(v)

    for bb in range(b):
        for hh in range(nh):
            var a = -exp(Float64(w.a_log[hh]))
            var h = List[Float64]()
            for i in range(p_dim * n_state):
                h.append(
                    Float64(
                        init_h[
                            ((bb * nh + hh) * p_dim) * n_state + i
                        ]
                    )
                )
            for li in range(l):
                var t = bb * l + li
                var dtv = dt[t * nh + hh]
                var da = exp(dtv * a)
                for p in range(p_dim):
                    var xv = xbc[t * cd + hh * p_dim + p] * dtv
                    for n in range(n_state):
                        var bv = xbc[t * cd + di + n]
                        h[p * n_state + n] = da * h[p * n_state + n] + bv * xv
                for p in range(p_dim):
                    var y = Float64(0.0)
                    for n in range(n_state):
                        y += xbc[t * cd + di + n_state + n] * h[p * n_state + n]
                    _set64(
                        out.skip_out,
                        (t * nh + hh) * p_dim + p,
                        y
                        + xbc[t * cd + hh * p_dim + p] * Float64(w.d_skip[hh]),
                        m * nh * p_dim,
                    )

    for t in range(m):
        # gated norm then out_proj then residual.
        var gate = List[Float64]()
        var acc = Float64(0.0)
        for j in range(di):
            var z = proj[t * dip + j]
            var g = out.skip_out[t * di + j] * (z / (1.0 + exp(-z)))
            gate.append(g)
            acc += g * g
        var rstd = 1.0 / sqrt(acc / Float64(di) + Float64(M2_RMS_EPS))
        for c in range(dm):
            var s = Float64(0.0)
            for j in range(di):
                s += (
                    Float64(w.gnorm_w[j]) * (gate[j] * rstd)
                ) * Float64(w.w_out[c * di + j])
            out.residual_out.append(Float64(x[t * dm + c]) + s)
    return out^


def _set64(mut xs: List[Float64], i: Int, v: Float64, size: Int):
    while len(xs) < size:
        xs.append(0.0)
    xs[i] = v
