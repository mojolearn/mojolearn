# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`mamba_ssm/modules/mamba3.py::Mamba3.forward` (:160-278, SISO arm at
the profile defaults), `::step` (:314-440, SEMANTICS only) and
`::allocate_inference_cache` (:442-482, the FOUR state pieces),
state-spaces/mamba `e9594ce`; residual/norm wrapper
`mamba_ssm/modules/block.py::Block.forward` non-fused arm (:51-53, :67 --
the in-repo citation; HF transformers @ d56c55b has NO mamba3 model) --
ON THE DEVICE, under profile `mojolearn.identical.mamba3.siso.fp32.v1`
(`mamba/IDENTICAL_MAMBA3_CONTRACT.md`). **COPY, DO NOT IMPROVE.** Seams
S1-S6, S21 and S23, composed around the SISO core
(`mamba/impl/mamba_ssm/ops/mamba3_siso.mojo`, S7-S20 + S22 + reports).

THE ENTRY POINT is `mamba3_block_forward`. Prefill is a fresh zero
`Mamba3DeviceState`; the decode step is the SAME function at `l == 1`
carrying the state (DEVIATION 831: decode is PREFILL RESUMPTION; the
carried state is theta, the SEALED boundary h and the last working
chunk's buffered rows -- DEVIATION 832, the oracle's docstring is the
authority). ONE spelling for both paths is what makes gate (d) a theorem
the gate verifies. The upstream per-token recurrence
(`mamba3_siso_step_ref`:99-146; the CuteDSL `mamba3_step_fn`, "Only
tested on H100", mamba3.py:320) rounds differently BY CONSTRUCTION and is
kept in this file ONLY as the required-RED arm STEP_UPSTREAM_RECURRENCE.

DELTAS FROM THE MAMBA-2 BLOCK, in one breath (contract section 0): NO
conv (S6/S7 of the siblings are GONE -- in_proj feeds the core directly,
and v is the RAW x split); NO dt clamp (S6 is bias -> softplus and
nothing else; do not import mamba2 S9's clamp); A is DATA-DEPENDENT per
(token, head) through the heavy-tail activation WITH the A_floor clamp
(S5 -- `identical_clamp` used whole, DEVIATION 788's primitive, its lower
bound -inf never binding); the gate is applied RAW inside the core (S19,
`is_outproj_norm=False`; there is no gated output norm); the B/C RMSNorms
(S21) are new, per (token, group), eps 1e-5, learned weight, NO bias --
the per-head biases are S12's, inside the core, AFTER the norm.

INHERITED MACHINERY, REUSED, NOT RESPELLED: S1-S3 through the Mamba-1
device kernel `modeling_mamba.mojo::mamba_rms_norm` VERBATIM; S23 through
its `residual_add_kernel`; upload/download/zeros and the refusal walk
through its helpers; both projections through
`gemm_identical.mojo::identical_gemm` (OP_NT in gemm_oracle's numbering
-- the orientation trap is that file's header note). The S21 fold below
is a NEW TRANSCRIPTION of the S1-S3 machinery (the Mamba-1 kernel
hard-addresses whole [M, width] rows and writes a sumsq stage; S21's two
norms per token at column offsets, with NO carded sumsq, need their own
addressing -- only the ADDRESSING differs, the arithmetic is character
for character the same chain).

REFUSED BY NAME, structurally: this surface carries no `is_mimo`,
`mimo_rank`, `is_outproj_norm`, `fuse_pregate_headwise_norm`,
`rope_fraction`, `ngroups`, varlen or dtype knob -- the profile constants
are compile-time facts and non-Float32 never enters (the shipped
surface's bf16 casts, mamba3_siso_combined.py:390-399, are REFUSED, not
reproduced). dt_min/dt_max/dt_init_floor are INITIALIZATION facts
(mamba3.py:111-115) and do not exist here.

THE SABOTAGE ARMS THIS FILE OWNS (contract 8f):

    MOJOLEARN_MAMBA3_SABOTAGE_A_FLOOR_UNCLAMPED       S5's clamp dropped
    MOJOLEARN_MAMBA3_SABOTAGE_STEP_UPSTREAM_RECURRENCE  step_ref:119-137 in decode

The step arm ENGAGES ONLY AT l == 1 (the mamba2 lesson, kept verbatim: an
armed decode gate legitimately runs a prefill leg through this entry
point first, and the arm must leave it untouched). While armed and
engaged it uses the state's `pend_k`/`pend_v` buffers as the step_ref's
K_State/V_State (they are otherwise idle outside an Input_States
continuation) and runs in three kernels so no thread reads a buffer
another thread of the same launch writes.

EVERYTHING HERE IS RUN OWED; no kernel has compiled or run. Commands live
in `mamba/checks/mamba3_check.mojo`.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace
from gemm.checks.gemm_identical import identical_gemm

# ORIENTATION NUMBERING: gemm_oracle's OP_NT = 1 -- the numbering
# identical_gemm reads. The Mamba-1 header's trap note applies verbatim.
from gemm.checks.gemm_oracle import OP_NT

from checks.numerics import (
    ftz,
    identical_clamp,
    identical_div,
    identical_exp,
    identical_mul_add,
    identical_rsqrt,
    identical_sigmoid,
    identical_softplus,
    identical_tanh,
    portable_cosf,
    portable_sinf,
)
from mamba.checks.mamba3_fixture import (
    BITS_POS_INF,
    M3_A_FLOOR,
    M3_CHUNK_SIZE,
    M3_D_STATE,
    M3_HEADDIM,
    M3_NUM_ROPE_ANGLES,
    M3_PI,
    M3_RMS_EPS,
    Mamba3Dims,
    Mamba3Weights,
    f32_from_bits,
)
from mamba.impl.mamba_ssm.ops.mamba3_siso import (
    MAMBA3_TPB,
    SISO3_ANY_SABOTAGE,
    m3_mod_2pi,
    m3_n_chunks,
    m3_q_eff,
    m3_siso_forward,
    siso3_sabotage_name,
)
from mamba.impl.transformers.models.mamba.modeling_mamba import (
    _refuse_nonfinite_named,
    mamba_download,
    mamba_rms_norm,
    mamba_upload,
    mamba_zeros,
    residual_add_kernel,
)


def pinned_mul(a: Float32, b: Float32) -> Float32:
    """DEVIATION 720's construction; see the oracle."""
    return identical_mul_add(a, b, Float32(-0.0))


def _grid(n: Int) -> Int:
    var g = (n + MAMBA3_TPB - 1) // MAMBA3_TPB
    if g < 1:
        return 1
    return g


def m3_neg_inf() -> Float32:
    return -f32_from_bits(BITS_POS_INF)


# ===========================================================================
# Sabotage arms (see header).
# ===========================================================================

comptime SAB3_A_FLOOR_UNCLAMPED = is_defined[
    "MOJOLEARN_MAMBA3_SABOTAGE_A_FLOOR_UNCLAMPED"
]()
comptime SAB3_STEP_UPSTREAM_RECURRENCE = is_defined[
    "MOJOLEARN_MAMBA3_SABOTAGE_STEP_UPSTREAM_RECURRENCE"
]()

comptime BLOCK3_ANY_SABOTAGE = (
    SAB3_A_FLOOR_UNCLAMPED
    or SAB3_STEP_UPSTREAM_RECURRENCE
    or SISO3_ANY_SABOTAGE
)


def mamba3_sabotage_name() -> String:
    comptime if SAB3_A_FLOOR_UNCLAMPED:
        return String("A_FLOOR_UNCLAMPED")
    comptime if SAB3_STEP_UPSTREAM_RECURRENCE:
        return String("STEP_UPSTREAM_RECURRENCE")
    return siso3_sabotage_name()


# ===========================================================================
# Device-side parameters, state and stages
# ===========================================================================


struct Mamba3DeviceWeights(Movable):
    """One block's parameters on the device, upstream shapes
    (`Mamba3Weights`'s table), row-major, contiguous. `weights_checked` is
    DEVIATION 1886's cache: the nine weight names are refusal-walked
    once, x and the state on every call."""

    var dims: Mamba3Dims
    var weights_checked: Bool
    var norm_w: DeviceBuffer[DType.float32]  # [d_model]
    var w_in: DeviceBuffer[DType.float32]  # [d_in_proj, d_model]
    var dt_bias: DeviceBuffer[DType.float32]  # [H]
    var bnorm_w: DeviceBuffer[DType.float32]  # [N]
    var cnorm_w: DeviceBuffer[DType.float32]  # [N]
    var b_bias: DeviceBuffer[DType.float32]  # [H, N]
    var c_bias: DeviceBuffer[DType.float32]  # [H, N]
    var d_skip: DeviceBuffer[DType.float32]  # [H]
    var w_out: DeviceBuffer[DType.float32]  # [d_model, d_inner]

    def __init__(out self, ctx: DeviceContext, w: Mamba3Weights) raises:
        self.dims = w.dims.copy()
        self.weights_checked = False
        self.norm_w = mamba_upload(ctx, w.norm_w)
        self.w_in = mamba_upload(ctx, w.w_in)
        self.dt_bias = mamba_upload(ctx, w.dt_bias)
        self.bnorm_w = mamba_upload(ctx, w.bnorm_w)
        self.cnorm_w = mamba_upload(ctx, w.cnorm_w)
        self.b_bias = mamba_upload(ctx, w.b_bias)
        self.c_bias = mamba_upload(ctx, w.c_bias)
        self.d_skip = mamba_upload(ctx, w.d_skip)
        self.w_out = mamba_upload(ctx, w.w_out)


struct Mamba3DeviceState(Movable):
    """DEVIATION 832's carried state on the device (the oracle's
    `Mamba3State`, same layouts): theta [B, H, R]; the sealed boundary h
    [B, H, P, N]; the last working chunk's buffered rows (`buf_len` of
    them, rotated-unscaled q/k [B, Q, H, N], raw v [B, Q, H, P], dt /
    sigma(trap) / ADT [B, Q, H]); and the pending Input_States pieces
    `pend_k` [B, H, N] / `pend_v` [B, H, P] (which double as the armed
    step recurrence's K_State/V_State). Zeros before the first token."""

    var b: Int
    var dims: Mamba3Dims
    var buf_len: Int
    var pending: Bool
    var theta: DeviceBuffer[DType.float32]
    var h: DeviceBuffer[DType.float32]
    var buf_qrot: DeviceBuffer[DType.float32]
    var buf_krot: DeviceBuffer[DType.float32]
    var buf_v: DeviceBuffer[DType.float32]
    var buf_dt: DeviceBuffer[DType.float32]
    var buf_sig: DeviceBuffer[DType.float32]
    var buf_adt: DeviceBuffer[DType.float32]
    var pend_k: DeviceBuffer[DType.float32]
    var pend_v: DeviceBuffer[DType.float32]

    def __init__(
        out self, ctx: DeviceContext, b: Int, dims: Mamba3Dims
    ) raises:
        self.b = b
        self.dims = dims.copy()
        self.buf_len = 0
        self.pending = False
        var nh = dims.nheads
        self.theta = mamba_zeros(ctx, b * nh * M3_NUM_ROPE_ANGLES)
        self.h = mamba_zeros(ctx, b * nh * M3_HEADDIM * M3_D_STATE)
        self.buf_qrot = mamba_zeros(
            ctx, b * M3_CHUNK_SIZE * nh * M3_D_STATE
        )
        self.buf_krot = mamba_zeros(
            ctx, b * M3_CHUNK_SIZE * nh * M3_D_STATE
        )
        self.buf_v = mamba_zeros(ctx, b * M3_CHUNK_SIZE * nh * M3_HEADDIM)
        self.buf_dt = mamba_zeros(ctx, b * M3_CHUNK_SIZE * nh)
        self.buf_sig = mamba_zeros(ctx, b * M3_CHUNK_SIZE * nh)
        self.buf_adt = mamba_zeros(ctx, b * M3_CHUNK_SIZE * nh)
        self.pend_k = mamba_zeros(ctx, b * nh * M3_D_STATE)
        self.pend_v = mamba_zeros(ctx, b * nh * M3_HEADDIM)

    def set_input_states(
        mut self,
        ctx: DeviceContext,
        theta_in: List[Float32],
        h_in: List[Float32],
        k_in: List[Float32],
        v_in: List[Float32],
    ) raises:
        """The upstream four-piece continuation (contract section 5 claim
        2); fresh state only; k/v held pending for S22."""
        if self.buf_len != 0:
            raise Error(
                "Mamba3DeviceState.set_input_states: the state is"
                " mid-sequence (buf_len = "
                + String(self.buf_len)
                + "); Input_States only has upstream meaning on a fresh"
                " state"
            )
        var nh = self.dims.nheads
        if len(theta_in) != self.b * nh * M3_NUM_ROPE_ANGLES:
            raise Error("set_input_states: theta size mismatch")
        if len(h_in) != self.b * nh * M3_HEADDIM * M3_D_STATE:
            raise Error("set_input_states: h size mismatch")
        if len(k_in) != self.b * nh * M3_D_STATE:
            raise Error("set_input_states: k size mismatch")
        if len(v_in) != self.b * nh * M3_HEADDIM:
            raise Error("set_input_states: v size mismatch")
        var tb = mamba_upload(ctx, theta_in)
        var hb = mamba_upload(ctx, h_in)
        var kb = mamba_upload(ctx, k_in)
        var vb = mamba_upload(ctx, v_in)
        ctx.enqueue_copy(dst_buf=self.theta, src_buf=tb)
        ctx.enqueue_copy(dst_buf=self.h, src_buf=hb)
        ctx.enqueue_copy(dst_buf=self.pend_k, src_buf=kb)
        ctx.enqueue_copy(dst_buf=self.pend_v, src_buf=vb)
        ctx.synchronize()
        _ = tb^
        _ = hb^
        _ = kb^
        _ = vb^
        self.pending = True


def allocate_inference_cache(
    ctx: DeviceContext, batch_size: Int, dims: Mamba3Dims
) raises -> Mamba3DeviceState:
    """Their :442-482: ZERO state (`max_seqlen`/dtype/device dropped for
    mamba1 DEVIATION 734's reasons); the buffered rows are DEVIATION
    832's addition and are zeros too."""
    return Mamba3DeviceState(ctx, batch_size, dims)


struct Mamba3DeviceStages(Movable):
    """Every recorded stage of one block call plus the working buffers.
    Token stages are [M = B*l, ...] over the NEW tokens; working buffers
    are [B, T = q0+l, ...] and the card records their new-token slice
    (DEVIATION 832(ii)). Chunk-shaped stages cover the working chunks at
    `m3_q_eff()`."""

    var b: Int
    var l: Int
    var q0: Int
    var t_work: Int
    var nc: Int
    var dims: Mamba3Dims
    var norm_sumsq: DeviceBuffer[DType.float32]  # [M]
    var norm_out: DeviceBuffer[DType.float32]  # [M, dm]
    var in_proj: DeviceBuffer[DType.float32]  # [M, dip]
    var a_out: DeviceBuffer[DType.float32]  # [M, H]
    var dt_out: DeviceBuffer[DType.float32]  # [M, H]
    var adt_work: DeviceBuffer[DType.float32]  # [B, T, H]
    var sig_work: DeviceBuffer[DType.float32]  # [B, T, H]
    var dt_work: DeviceBuffer[DType.float32]  # [B, T, H]
    var gamma_work: DeviceBuffer[DType.float32]  # [B, T, H]
    var betap_work: DeviceBuffer[DType.float32]  # [B, T, H]
    var scale_work: DeviceBuffer[DType.float32]  # [B, T, H]
    var bcnorm_b: DeviceBuffer[DType.float32]  # [M, N]
    var bcnorm_c: DeviceBuffer[DType.float32]  # [M, N]
    var theta_out: DeviceBuffer[DType.float32]  # [M, H, R]
    var rotq_work: DeviceBuffer[DType.float32]  # [B, T, H, N]
    var rotk_work: DeviceBuffer[DType.float32]  # [B, T, H, N]
    var qkdot: DeviceBuffer[DType.float32]  # [M, H]
    var kscale_work: DeviceBuffer[DType.float32]  # [B, T, H, N]
    var v_work: DeviceBuffer[DType.float32]  # [B, T, H, P]
    var dacs: DeviceBuffer[DType.float32]  # [B, H, C, Q]
    var seg_l: DeviceBuffer[DType.float32]  # [B, C, H, Q, Q]
    var qk_s: DeviceBuffer[DType.float32]  # [B, C, H, Q, Q] scratch
    var pass_states: DeviceBuffer[DType.float32]  # [B, C, H, P, N]
    var yintra: DeviceBuffer[DType.float32]  # [M, H, P]
    var ystate: DeviceBuffer[DType.float32]  # [M, H, P]
    var skip_out: DeviceBuffer[DType.float32]  # [M, H, P]
    var gate_out: DeviceBuffer[DType.float32]  # [M, H, P]
    var out_proj: DeviceBuffer[DType.float32]  # [M, dm]
    var residual_out: DeviceBuffer[DType.float32]  # [M, dm]
    var h_last: DeviceBuffer[DType.float32]  # [B, H, P, N]
    var k_last: DeviceBuffer[DType.float32]  # [B, H, N]
    var v_last: DeviceBuffer[DType.float32]  # [B, H, P]
    var theta_last: DeviceBuffer[DType.float32]  # [B, H, R]

    def __init__(
        out self,
        ctx: DeviceContext,
        b: Int,
        l: Int,
        q0: Int,
        dims: Mamba3Dims,
    ) raises:
        self.b = b
        self.l = l
        self.q0 = q0
        self.t_work = q0 + l
        self.nc = m3_n_chunks(self.t_work)
        self.dims = dims.copy()
        var m = b * l
        var t = self.t_work
        var nc = self.nc
        var qv = m3_q_eff()
        var dm = dims.d_model
        var dip = dims.d_in_proj()
        var nh = dims.nheads
        comptime p_dim = M3_HEADDIM
        comptime n_state = M3_D_STATE
        comptime r_ang = M3_NUM_ROPE_ANGLES
        self.norm_sumsq = mamba_zeros(ctx, m)
        self.norm_out = mamba_zeros(ctx, m * dm)
        self.in_proj = mamba_zeros(ctx, m * dip)
        self.a_out = mamba_zeros(ctx, m * nh)
        self.dt_out = mamba_zeros(ctx, m * nh)
        self.adt_work = mamba_zeros(ctx, b * t * nh)
        self.sig_work = mamba_zeros(ctx, b * t * nh)
        self.dt_work = mamba_zeros(ctx, b * t * nh)
        self.gamma_work = mamba_zeros(ctx, b * t * nh)
        self.betap_work = mamba_zeros(ctx, b * t * nh)
        self.scale_work = mamba_zeros(ctx, b * t * nh)
        self.bcnorm_b = mamba_zeros(ctx, m * n_state)
        self.bcnorm_c = mamba_zeros(ctx, m * n_state)
        self.theta_out = mamba_zeros(ctx, m * nh * r_ang)
        self.rotq_work = mamba_zeros(ctx, b * t * nh * n_state)
        self.rotk_work = mamba_zeros(ctx, b * t * nh * n_state)
        self.qkdot = mamba_zeros(ctx, m * nh)
        self.kscale_work = mamba_zeros(ctx, b * t * nh * n_state)
        self.v_work = mamba_zeros(ctx, b * t * nh * p_dim)
        self.dacs = mamba_zeros(ctx, b * nh * nc * qv)
        self.seg_l = mamba_zeros(ctx, b * nc * nh * qv * qv)
        self.qk_s = mamba_zeros(ctx, b * nc * nh * qv * qv)
        self.pass_states = mamba_zeros(ctx, b * nc * nh * p_dim * n_state)
        self.yintra = mamba_zeros(ctx, m * nh * p_dim)
        self.ystate = mamba_zeros(ctx, m * nh * p_dim)
        self.skip_out = mamba_zeros(ctx, m * nh * p_dim)
        self.gate_out = mamba_zeros(ctx, m * nh * p_dim)
        self.out_proj = mamba_zeros(ctx, m * dm)
        self.residual_out = mamba_zeros(ctx, m * dm)
        self.h_last = mamba_zeros(ctx, b * nh * p_dim * n_state)
        self.k_last = mamba_zeros(ctx, b * nh * n_state)
        self.v_last = mamba_zeros(ctx, b * nh * p_dim)
        self.theta_last = mamba_zeros(ctx, b * nh * r_ang)


# ===========================================================================
# S5 (data-dependent A: heavy-tail + A_floor clamp) + S6 (dt: bias ->
# softplus, NO clamp). One thread per (b, li, h). Owns A_FLOOR_UNCLAMPED.
# ===========================================================================


def m3_a_dt_kernel(
    a_out: MutPointer[Float32, MutAnyOrigin],  # [M, H]
    dt_out: MutPointer[Float32, MutAnyOrigin],  # [M, H]
    in_proj: MutPointer[Float32, MutAnyOrigin],  # [M, dip]
    dt_bias: MutPointer[Float32, MutAnyOrigin],  # [H]
    n_in: Int32,  # M * H
    nh_in: Int32,
    dip_in: Int32,
    c_dt_in: Int32,
    c_a_in: Int32,
    neg_inf: Float32,
    neg_floor: Float32,  # -A_floor
):
    var n = Int(n_in)
    var nh = Int(nh_in)
    var dip = Int(dip_in)
    var c_dt = Int(c_dt_in)
    var c_a = Int(c_a_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= n:
        return
    var t = cell // nh
    var hh = cell - t * nh
    # S5: piecewise heavy-tail (contract S5's branch spelling, bit-equal
    # to the reference's branchless sum), negate exact, clamp.
    var xa = ftz(in_proj.unsafe_load(t * dip + c_a + hh))
    var ht: Float32
    if xa >= Float32(0.0):
        ht = ftz(Float32(1.0) + xa)
    else:
        ht = ftz(identical_div(Float32(1.0), ftz(Float32(1.0) - xa)))
    comptime if SAB3_A_FLOOR_UNCLAMPED:
        # SABOTAGE: the S5 clamp dropped. Witnessed ONLY by planted
        # dd_A < -9999 cells (the fixture's m3_adv_a_floor case).
        a_out.unsafe_store(cell, -ht)
    else:
        a_out.unsafe_store(
            cell, ftz(identical_clamp(-ht, neg_inf, neg_floor))
        )
    # S6: dt = softplus(dd_dt + dt_bias) -- and NOTHING else.
    var biased = ftz(
        ftz(in_proj.unsafe_load(t * dip + c_dt + hh))
        + ftz(dt_bias.unsafe_load(hh))
    )
    dt_out.unsafe_store(cell, ftz(identical_softplus(biased)))


# ===========================================================================
# S21 -- the B and C RMSNorms over d_state per token (G = 1): the S1-S3
# fold/rstd/product machinery at eps 1e-5, learned weight, NO gate, NO
# bias. One thread per token computing both rows.
# ===========================================================================


def m3_bcnorm_kernel(
    bcb: MutPointer[Float32, MutAnyOrigin],  # [M, N]
    bcc: MutPointer[Float32, MutAnyOrigin],  # [M, N]
    in_proj: MutPointer[Float32, MutAnyOrigin],  # [M, dip]
    bnorm_w: MutPointer[Float32, MutAnyOrigin],  # [N]
    cnorm_w: MutPointer[Float32, MutAnyOrigin],  # [N]
    m_in: Int32,
    dip_in: Int32,
    c_b_in: Int32,
    c_c_in: Int32,
    eps: Float32,
):
    comptime n_state = M3_D_STATE
    var m = Int(m_in)
    var dip = Int(dip_in)
    var c_b = Int(c_b_in)
    var c_c = Int(c_c_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= m:
        return
    var accb = Float32(0.0)
    var accc = Float32(0.0)
    for n in range(n_state):
        var bj = ftz(in_proj.unsafe_load(t * dip + c_b + n))
        accb = ftz(identical_mul_add(bj, bj, accb))
        var cj = ftz(in_proj.unsafe_load(t * dip + c_c + n))
        accc = ftz(identical_mul_add(cj, cj, accc))
    var rstdb = ftz(
        identical_rsqrt(
            ftz(ftz(identical_div(accb, Float32(n_state))) + eps)
        )
    )
    var rstdc = ftz(
        identical_rsqrt(
            ftz(ftz(identical_div(accc, Float32(n_state))) + eps)
        )
    )
    for n in range(n_state):
        var innerb = ftz(
            pinned_mul(ftz(in_proj.unsafe_load(t * dip + c_b + n)), rstdb)
        )
        bcb.unsafe_store(
            t * n_state + n,
            ftz(pinned_mul(ftz(bnorm_w.unsafe_load(n)), innerb)),
        )
        var innerc = ftz(
            pinned_mul(ftz(in_proj.unsafe_load(t * dip + c_c + n)), rstdc)
        )
        bcc.unsafe_store(
            t * n_state + n,
            ftz(pinned_mul(ftz(cnorm_w.unsafe_load(n)), innerc)),
        )


# ===========================================================================
# Working-sequence assembly (copies, not seams): buffered rows into the
# working arrays' rows [0, q0); the NEW rows' raw v (the x split) into
# rows [q0, T).
# ===========================================================================


def m3_assemble_buf_kernel(
    rotq_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, H, N]
    rotk_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, H, N]
    v_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, H, P]
    dt_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, H]
    sig_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, H]
    adt_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, H]
    buf_qrot: MutPointer[Float32, MutAnyOrigin],  # [B, Q, H, N]
    buf_krot: MutPointer[Float32, MutAnyOrigin],  # [B, Q, H, N]
    buf_v: MutPointer[Float32, MutAnyOrigin],  # [B, Q, H, P]
    buf_dt: MutPointer[Float32, MutAnyOrigin],  # [B, Q, H]
    buf_sig: MutPointer[Float32, MutAnyOrigin],  # [B, Q, H]
    buf_adt: MutPointer[Float32, MutAnyOrigin],  # [B, Q, H]
    b_in: Int32,
    l_in: Int32,
    q0_in: Int32,
    nh_in: Int32,
    q_cap_in: Int32,
):
    comptime n_state = M3_D_STATE
    comptime p_dim = M3_HEADDIM
    var b = Int(b_in)
    var l = Int(l_in)
    var q0 = Int(q0_in)
    var nh = Int(nh_in)
    var q_cap = Int(q_cap_in)
    var t_work = q0 + l
    # per-row width across the six arrays: N + N + P + 3, per head.
    var wrow = nh * (2 * n_state + p_dim + 3)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= b * q0 * wrow:
        return
    var bb = cell // (q0 * wrow)
    var rem = cell - bb * q0 * wrow
    var t = rem // wrow
    var d = rem - t * wrow
    var hh = d // (2 * n_state + p_dim + 3)
    var e = d - hh * (2 * n_state + p_dim + 3)
    var widx3 = (bb * t_work + t) * nh + hh
    var bidx3 = (bb * q_cap + t) * nh + hh
    if e < n_state:
        rotq_work.unsafe_store(
            widx3 * n_state + e, buf_qrot.unsafe_load(bidx3 * n_state + e)
        )
    elif e < 2 * n_state:
        var n = e - n_state
        rotk_work.unsafe_store(
            widx3 * n_state + n, buf_krot.unsafe_load(bidx3 * n_state + n)
        )
    elif e < 2 * n_state + p_dim:
        var p = e - 2 * n_state
        v_work.unsafe_store(
            widx3 * p_dim + p, buf_v.unsafe_load(bidx3 * p_dim + p)
        )
    elif e == 2 * n_state + p_dim:
        dt_work.unsafe_store(widx3, buf_dt.unsafe_load(bidx3))
    elif e == 2 * n_state + p_dim + 1:
        sig_work.unsafe_store(widx3, buf_sig.unsafe_load(bidx3))
    else:
        adt_work.unsafe_store(widx3, buf_adt.unsafe_load(bidx3))


def m3_assemble_vnew_kernel(
    v_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, H, P]
    in_proj: MutPointer[Float32, MutAnyOrigin],  # [M, dip]
    b_in: Int32,
    l_in: Int32,
    q0_in: Int32,
    nh_in: Int32,
    dip_in: Int32,
    c_x_in: Int32,
):
    comptime p_dim = M3_HEADDIM
    var b = Int(b_in)
    var l = Int(l_in)
    var q0 = Int(q0_in)
    var nh = Int(nh_in)
    var dip = Int(dip_in)
    var c_x = Int(c_x_in)
    var t_work = q0 + l
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= b * l * nh * p_dim:
        return
    var bb = cell // (l * nh * p_dim)
    var rem = cell - bb * l * nh * p_dim
    var li = rem // (nh * p_dim)
    var d = rem - li * nh * p_dim
    v_work.unsafe_store(
        ((bb * t_work + q0 + li) * nh) * p_dim + d,
        in_proj.unsafe_load((bb * l + li) * dip + c_x + d),
    )


# ===========================================================================
# Buffer update (copies): keep the LAST WORKING CHUNK's r = t_work -
# (C-1)*Q rows (DEVIATION 832(i); r in [1, Q], the buffer never empties).
# ===========================================================================


def m3_buffer_update_kernel(
    buf_qrot: MutPointer[Float32, MutAnyOrigin],
    buf_krot: MutPointer[Float32, MutAnyOrigin],
    buf_v: MutPointer[Float32, MutAnyOrigin],
    buf_dt: MutPointer[Float32, MutAnyOrigin],
    buf_sig: MutPointer[Float32, MutAnyOrigin],
    buf_adt: MutPointer[Float32, MutAnyOrigin],
    rotq_work: MutPointer[Float32, MutAnyOrigin],
    rotk_work: MutPointer[Float32, MutAnyOrigin],
    v_work: MutPointer[Float32, MutAnyOrigin],
    dt_work: MutPointer[Float32, MutAnyOrigin],
    sig_work: MutPointer[Float32, MutAnyOrigin],
    adt_work: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32,
    t_in: Int32,
    r_in: Int32,
    nh_in: Int32,
    q_cap_in: Int32,
):
    comptime n_state = M3_D_STATE
    comptime p_dim = M3_HEADDIM
    var b = Int(b_in)
    var t_work = Int(t_in)
    var r = Int(r_in)
    var nh = Int(nh_in)
    var q_cap = Int(q_cap_in)
    var wrow = nh * (2 * n_state + p_dim + 3)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= b * r * wrow:
        return
    var bb = cell // (r * wrow)
    var rem = cell - bb * r * wrow
    var t = rem // wrow
    var d = rem - t * wrow
    var hh = d // (2 * n_state + p_dim + 3)
    var e = d - hh * (2 * n_state + p_dim + 3)
    var src = t_work - r + t
    var widx3 = (bb * t_work + src) * nh + hh
    var bidx3 = (bb * q_cap + t) * nh + hh
    if e < n_state:
        buf_qrot.unsafe_store(
            bidx3 * n_state + e, rotq_work.unsafe_load(widx3 * n_state + e)
        )
    elif e < 2 * n_state:
        var n = e - n_state
        buf_krot.unsafe_store(
            bidx3 * n_state + n, rotk_work.unsafe_load(widx3 * n_state + n)
        )
    elif e < 2 * n_state + p_dim:
        var p = e - 2 * n_state
        buf_v.unsafe_store(
            bidx3 * p_dim + p, v_work.unsafe_load(widx3 * p_dim + p)
        )
    elif e == 2 * n_state + p_dim:
        buf_dt.unsafe_store(bidx3, dt_work.unsafe_load(widx3))
    elif e == 2 * n_state + p_dim + 1:
        buf_sig.unsafe_store(bidx3, sig_work.unsafe_load(widx3))
    else:
        buf_adt.unsafe_store(bidx3, adt_work.unsafe_load(widx3))


# ===========================================================================
# THE REQUIRED-RED ARM: STEP_UPSTREAM_RECURRENCE (mamba3_siso_step_ref
# :99-146). The upstream torch step's own rounding: alpha = exp(ADT),
# beta = ((1 - sigma) * dt) * alpha, gamma = sigma * dt, the THREE-term
# in-place state update S = alpha*S + beta*(K_st (x) V_st) +
# gamma*(k_rot (x) v), readout q_rot . S AFTER the update, D last, and
# the z-gate spelled `(out * z) * sigmoid(z)` (:136 -- NOT the
# one-division silu). Exists to be falsified by gate (d); never the
# profile. Three kernels so no cross-thread read-after-write exists:
# angle advance (in place, same spelling as the profile's -- the two
# paths agree at S10), the core update/readout, then the K/V state store.
# ===========================================================================


def m3_step_angle_kernel(
    theta_state: MutPointer[Float32, MutAnyOrigin],  # [B, H, R] in/out
    in_proj: MutPointer[Float32, MutAnyOrigin],  # [B, dip] (l = 1)
    dt_out: MutPointer[Float32, MutAnyOrigin],  # [B, H]
    b_in: Int32,
    nh_in: Int32,
    dip_in: Int32,
    c_ang_in: Int32,
):
    comptime r_ang = M3_NUM_ROPE_ANGLES
    var b = Int(b_in)
    var nh = Int(nh_in)
    var dip = Int(dip_in)
    var c_ang = Int(c_ang_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= b * nh * r_ang:
        return
    var bb = cell // (nh * r_ang)
    var rem = cell - bb * nh * r_ang
    var hh = rem // r_ang
    var r = rem - hh * r_ang
    var a = ftz(
        pinned_mul(
            identical_tanh(ftz(in_proj.unsafe_load(bb * dip + c_ang + r))),
            M3_PI,
        )
    )
    var inc = ftz(pinned_mul(a, ftz(dt_out.unsafe_load(bb * nh + hh))))
    theta_state.unsafe_store(
        cell, m3_mod_2pi(ftz(ftz(theta_state.unsafe_load(cell)) + inc))
    )


def m3_step_core_kernel(
    skip_out: MutPointer[Float32, MutAnyOrigin],  # [B, H, P]
    gate_out: MutPointer[Float32, MutAnyOrigin],  # [B, H, P]
    h_state: MutPointer[Float32, MutAnyOrigin],  # [B, H, P, N] in/out
    theta_state: MutPointer[Float32, MutAnyOrigin],  # [B, H, R] (advanced)
    pend_k: MutPointer[Float32, MutAnyOrigin],  # [B, H, N] = K_State (old)
    pend_v: MutPointer[Float32, MutAnyOrigin],  # [B, H, P] = V_State (old)
    in_proj: MutPointer[Float32, MutAnyOrigin],  # [B, dip]
    a_out: MutPointer[Float32, MutAnyOrigin],  # [B, H]
    dt_out: MutPointer[Float32, MutAnyOrigin],  # [B, H]
    bcb: MutPointer[Float32, MutAnyOrigin],  # [B, N]
    bcc: MutPointer[Float32, MutAnyOrigin],  # [B, N]
    b_bias: MutPointer[Float32, MutAnyOrigin],  # [H, N]
    c_bias: MutPointer[Float32, MutAnyOrigin],  # [H, N]
    d_skip: MutPointer[Float32, MutAnyOrigin],  # [H]
    b_in: Int32,
    nh_in: Int32,
    dip_in: Int32,
    c_z_in: Int32,
    c_x_in: Int32,
    c_trap_in: Int32,
):
    comptime n_state = M3_D_STATE
    comptime p_dim = M3_HEADDIM
    comptime r_ang = M3_NUM_ROPE_ANGLES
    var b = Int(b_in)
    var nh = Int(nh_in)
    var dip = Int(dip_in)
    var c_z = Int(c_z_in)
    var c_x = Int(c_x_in)
    var c_trap = Int(c_trap_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= b * nh * p_dim:
        return
    var bb = cell // (nh * p_dim)
    var rem = cell - bb * nh * p_dim
    var hh = rem // p_dim
    var p = rem - hh * p_dim
    var dtv = ftz(dt_out.unsafe_load(bb * nh + hh))
    var adt = ftz(pinned_mul(ftz(a_out.unsafe_load(bb * nh + hh)), dtv))
    var sg = ftz(
        identical_sigmoid(ftz(in_proj.unsafe_load(bb * dip + c_trap + hh)))
    )
    var al = ftz(identical_exp(adt))
    # beta = ((1 - sigma) * dt) * alpha, torch's left association (:121).
    var be = ftz(
        pinned_mul(ftz(pinned_mul(ftz(Float32(1.0) - sg), dtv)), al)
    )
    var ga = ftz(pinned_mul(sg, dtv))
    var vv = ftz(in_proj.unsafe_load(bb * dip + c_x + hh * p_dim + p))
    var vst = ftz(pend_v.unsafe_load((bb * nh + hh) * p_dim + p))
    var out = Float32(0.0)
    for n in range(n_state):
        # k_rot[n], q_rot[n] recomputed per element from the normed B/C,
        # the biases and the ADVANCED theta (pair partner recomputed too).
        var pair = n // 2
        var kn: Float32
        var qn: Float32
        var b0 = ftz(
            ftz(bcb.unsafe_load(bb * n_state + n))
            + ftz(b_bias.unsafe_load(hh * n_state + n))
        )
        var q0v = ftz(
            ftz(bcc.unsafe_load(bb * n_state + n))
            + ftz(c_bias.unsafe_load(hh * n_state + n))
        )
        if pair < r_ang:
            var th = ftz(
                theta_state.unsafe_load((bb * nh + hh) * r_ang + pair)
            )
            var cv = ftz(portable_cosf(th))
            var sv = ftz(portable_sinf(th))
            var e0 = 2 * pair
            var e1 = e0 + 1
            var bx0 = ftz(
                ftz(bcb.unsafe_load(bb * n_state + e0))
                + ftz(b_bias.unsafe_load(hh * n_state + e0))
            )
            var bx1 = ftz(
                ftz(bcb.unsafe_load(bb * n_state + e1))
                + ftz(b_bias.unsafe_load(hh * n_state + e1))
            )
            var qx0 = ftz(
                ftz(bcc.unsafe_load(bb * n_state + e0))
                + ftz(c_bias.unsafe_load(hh * n_state + e0))
            )
            var qx1 = ftz(
                ftz(bcc.unsafe_load(bb * n_state + e1))
                + ftz(c_bias.unsafe_load(hh * n_state + e1))
            )
            if n == e0:
                kn = ftz(pinned_mul(bx0, cv) - pinned_mul(bx1, sv))
                qn = ftz(pinned_mul(qx0, cv) - pinned_mul(qx1, sv))
            else:
                kn = ftz(pinned_mul(bx0, sv) + pinned_mul(bx1, cv))
                qn = ftz(pinned_mul(qx0, sv) + pinned_mul(qx1, cv))
        else:
            kn = b0
            qn = q0v
        var hidx = ((bb * nh + hh) * p_dim + p) * n_state + n
        var kst = ftz(pend_k.unsafe_load((bb * nh + hh) * n_state + n))
        # S = alpha*S + beta*(K_st*V_st) + gamma*(k*v): torch's three
        # separate roundings (:125-127), k*v product FIRST inside each.
        var s1 = ftz(pinned_mul(al, ftz(h_state.unsafe_load(hidx))))
        var s2 = ftz(s1 + ftz(pinned_mul(be, ftz(pinned_mul(kst, vst)))))
        var s3 = ftz(s2 + ftz(pinned_mul(ga, ftz(pinned_mul(kn, vv)))))
        h_state.unsafe_store(hidx, s3)
        # readout AFTER the update (:130).
        out = ftz(identical_mul_add(s3, qn, out))
    out = ftz(out)
    # D last (:133).
    out = ftz(out + ftz(pinned_mul(ftz(d_skip.unsafe_load(hh)), vv)))
    skip_out.unsafe_store(cell, out)
    # the z gate, torch's (out * z) * sigmoid(z) (:136) -- NOT the
    # one-division silu; part of what makes this arm RED.
    var zv = ftz(in_proj.unsafe_load(bb * dip + c_z + hh * p_dim + p))
    gate_out.unsafe_store(
        cell,
        ftz(pinned_mul(ftz(pinned_mul(out, zv)), identical_sigmoid(zv))),
    )


def m3_step_state_kernel(
    pend_k: MutPointer[Float32, MutAnyOrigin],  # [B, H, N] out
    pend_v: MutPointer[Float32, MutAnyOrigin],  # [B, H, P] out
    theta_state: MutPointer[Float32, MutAnyOrigin],  # [B, H, R] (advanced)
    in_proj: MutPointer[Float32, MutAnyOrigin],  # [B, dip]
    bcb: MutPointer[Float32, MutAnyOrigin],  # [B, N]
    b_bias: MutPointer[Float32, MutAnyOrigin],  # [H, N]
    b_in: Int32,
    nh_in: Int32,
    dip_in: Int32,
    c_x_in: Int32,
):
    comptime n_state = M3_D_STATE
    comptime p_dim = M3_HEADDIM
    comptime r_ang = M3_NUM_ROPE_ANGLES
    var b = Int(b_in)
    var nh = Int(nh_in)
    var dip = Int(dip_in)
    var c_x = Int(c_x_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= b * nh * (n_state + p_dim):
        return
    var bb = cell // (nh * (n_state + p_dim))
    var rem = cell - bb * nh * (n_state + p_dim)
    var hh = rem // (n_state + p_dim)
    var d = rem - hh * (n_state + p_dim)
    if d < n_state:
        var n = d
        var pair = n // 2
        var kn: Float32
        if pair < r_ang:
            var th = ftz(
                theta_state.unsafe_load((bb * nh + hh) * r_ang + pair)
            )
            var cv = ftz(portable_cosf(th))
            var sv = ftz(portable_sinf(th))
            var e0 = 2 * pair
            var e1 = e0 + 1
            var bx0 = ftz(
                ftz(bcb.unsafe_load(bb * n_state + e0))
                + ftz(b_bias.unsafe_load(hh * n_state + e0))
            )
            var bx1 = ftz(
                ftz(bcb.unsafe_load(bb * n_state + e1))
                + ftz(b_bias.unsafe_load(hh * n_state + e1))
            )
            if n == e0:
                kn = ftz(pinned_mul(bx0, cv) - pinned_mul(bx1, sv))
            else:
                kn = ftz(pinned_mul(bx0, sv) + pinned_mul(bx1, cv))
        else:
            kn = ftz(
                ftz(bcb.unsafe_load(bb * n_state + n))
                + ftz(b_bias.unsafe_load(hh * n_state + n))
            )
        pend_k.unsafe_store((bb * nh + hh) * n_state + n, kn)
    else:
        var p = d - n_state
        pend_v.unsafe_store(
            (bb * nh + hh) * p_dim + p,
            in_proj.unsafe_load(bb * dip + c_x + hh * p_dim + p),
        )


# ===========================================================================
# Refusal (contract section 6): every named input, BY BITS, before any
# recorded stage. Weight names cached after the first call (DEVIATION
# 1886's pattern); x and the state walked every call.
# ===========================================================================


def mamba3_refuse_bad_inputs(
    ctx: DeviceContext,
    mut w: Mamba3DeviceWeights,
    mut x: DeviceBuffer[DType.float32],
    mut state: Mamba3DeviceState,
    b: Int,
    l: Int,
) raises:
    var dims = w.dims.copy()
    var dm = dims.d_model
    var di = dims.d_inner
    var dip = dims.d_in_proj()
    var nh = dims.nheads
    _refuse_nonfinite_named("x", mamba_download(ctx, x, b * l * dm))
    if not w.weights_checked:
        _refuse_nonfinite_named(
            "norm.weight", mamba_download(ctx, w.norm_w, dm)
        )
        _refuse_nonfinite_named(
            "in_proj.weight", mamba_download(ctx, w.w_in, dip * dm)
        )
        _refuse_nonfinite_named(
            "dt_bias", mamba_download(ctx, w.dt_bias, nh)
        )
        _refuse_nonfinite_named(
            "B_norm.weight", mamba_download(ctx, w.bnorm_w, M3_D_STATE)
        )
        _refuse_nonfinite_named(
            "C_norm.weight", mamba_download(ctx, w.cnorm_w, M3_D_STATE)
        )
        _refuse_nonfinite_named(
            "B_bias", mamba_download(ctx, w.b_bias, nh * M3_D_STATE)
        )
        _refuse_nonfinite_named(
            "C_bias", mamba_download(ctx, w.c_bias, nh * M3_D_STATE)
        )
        _refuse_nonfinite_named("D", mamba_download(ctx, w.d_skip, nh))
        _refuse_nonfinite_named(
            "out_proj.weight", mamba_download(ctx, w.w_out, dm * di)
        )
        w.weights_checked = True
    _refuse_nonfinite_named(
        "state.theta",
        mamba_download(ctx, state.theta, b * nh * M3_NUM_ROPE_ANGLES),
    )
    _refuse_nonfinite_named(
        "state.h",
        mamba_download(ctx, state.h, b * nh * M3_HEADDIM * M3_D_STATE),
    )
    _refuse_nonfinite_named(
        "state.buf_qrot",
        mamba_download(
            ctx, state.buf_qrot, b * M3_CHUNK_SIZE * nh * M3_D_STATE
        ),
    )
    _refuse_nonfinite_named(
        "state.buf_krot",
        mamba_download(
            ctx, state.buf_krot, b * M3_CHUNK_SIZE * nh * M3_D_STATE
        ),
    )
    _refuse_nonfinite_named(
        "state.buf_v",
        mamba_download(
            ctx, state.buf_v, b * M3_CHUNK_SIZE * nh * M3_HEADDIM
        ),
    )
    _refuse_nonfinite_named(
        "state.buf_dt",
        mamba_download(ctx, state.buf_dt, b * M3_CHUNK_SIZE * nh),
    )
    _refuse_nonfinite_named(
        "state.buf_sig",
        mamba_download(ctx, state.buf_sig, b * M3_CHUNK_SIZE * nh),
    )
    _refuse_nonfinite_named(
        "state.buf_adt",
        mamba_download(ctx, state.buf_adt, b * M3_CHUNK_SIZE * nh),
    )
    if state.pending:
        _refuse_nonfinite_named(
            "input_states.k",
            mamba_download(ctx, state.pend_k, b * nh * M3_D_STATE),
        )
        _refuse_nonfinite_named(
            "input_states.v",
            mamba_download(ctx, state.pend_v, b * nh * M3_HEADDIM),
        )


# ===========================================================================
# Card recording helper (DEVIATION 832(ii): token-shaped SSD stages are
# recorded as the NEW-token slice of the working buffer; a copy).
# ===========================================================================


def _record_work_slice(
    ctx: DeviceContext,
    mut trace: IdentityTrace,
    tag: String,
    mut buf: DeviceBuffer[DType.float32],
    b: Int,
    l: Int,
    q0: Int,
    width: Int,
) raises:
    """Record rows [q0, q0+l) of a [B, T, width] working buffer as the
    [M, width] card stage."""
    var t_work = q0 + l
    var whole = mamba_download(ctx, buf, b * t_work * width)
    var out = List[Float32]()
    for bb in range(b):
        for li in range(l):
            for j in range(width):
                out.append(whole[((bb * t_work) + (q0 + li)) * width + j])
    trace.record_list_f32(tag, out)


# ===========================================================================
# THE ENTRY POINT
# ===========================================================================


def mamba3_block_forward(
    ctx: DeviceContext,
    mut stages: Mamba3DeviceStages,
    mut state: Mamba3DeviceState,
    mut w: Mamba3DeviceWeights,
    mut x: DeviceBuffer[DType.float32],
    b: Int,
    l: Int,
    mut trace: IdentityTrace,
    prefix: String,
) raises:
    """One block call (Block.forward around the SISO mixer), every stage
    of contract section 7 recorded under `prefix` in the section's order.
    Prefill = fresh zero state; decode = same entry, state carried
    (DEVIATIONS 831/832)."""
    if b <= 0 or l <= 0:
        raise Error(
            "mamba3_block_forward: B and L must be positive, got B="
            + String(b)
            + " L="
            + String(l)
        )
    if stages.b != b or stages.l != l or stages.q0 != state.buf_len:
        raise Error(
            "mamba3_block_forward: stages were built for B="
            + String(stages.b)
            + " L="
            + String(stages.l)
            + " q0="
            + String(stages.q0)
            + " but the call is B="
            + String(b)
            + " L="
            + String(l)
            + " q0="
            + String(state.buf_len)
        )
    if state.b != b or state.dims.d_model != stages.dims.d_model:
        raise Error(
            "mamba3_block_forward: the state's shape is not the call's"
        )
    if w.dims.d_model != stages.dims.d_model:
        raise Error(
            "mamba3_block_forward: the weights' d_model is not the stages'"
        )

    mamba3_refuse_bad_inputs(ctx, w, x, state, b, l)

    var dims = stages.dims.copy()
    var dm = dims.d_model
    var di = dims.d_inner
    var dip = dims.d_in_proj()
    var nh = dims.nheads
    comptime p_dim = M3_HEADDIM
    comptime n_state = M3_D_STATE
    comptime r_ang = M3_NUM_ROPE_ANGLES
    var m = b * l
    var q0 = state.buf_len
    var t_work = q0 + l
    var qv = m3_q_eff()
    var nc = stages.nc

    trace.record_device[DType.float32](ctx, prefix + ".input.x", x, m * dm)

    # ---- S1-S3: the block RMSNorm, the REUSED Mamba-1 kernel.
    mamba_rms_norm(
        ctx, stages.norm_sumsq, stages.norm_out, x, w.norm_w, m, dm
    )
    ctx.synchronize()

    # ---- S4: in_proj (mamba3.py:176), gemm v1 OP_NT, k = d_model.
    identical_gemm(
        ctx, stages.in_proj, stages.norm_out, w.w_in, m, dip, dm, OP_NT
    )

    # ---- S5 + S6.
    ctx.enqueue_function[m3_a_dt_kernel](
        stages.a_out.unsafe_ptr(),
        stages.dt_out.unsafe_ptr(),
        stages.in_proj.unsafe_ptr(),
        w.dt_bias.unsafe_ptr(),
        Int32(m * nh),
        Int32(nh),
        Int32(dip),
        Int32(dims.col_dt()),
        Int32(dims.col_a()),
        m3_neg_inf(),
        -M3_A_FLOOR,
        grid_dim=(_grid(m * nh), 1, 1),
        block_dim=(MAMBA3_TPB, 1, 1),
    )

    # ---- S21: the B/C norms.
    ctx.enqueue_function[m3_bcnorm_kernel](
        stages.bcnorm_b.unsafe_ptr(),
        stages.bcnorm_c.unsafe_ptr(),
        stages.in_proj.unsafe_ptr(),
        w.bnorm_w.unsafe_ptr(),
        w.cnorm_w.unsafe_ptr(),
        Int32(m),
        Int32(dip),
        Int32(dims.col_b()),
        Int32(dims.col_c()),
        M3_RMS_EPS,
        grid_dim=(_grid(m), 1, 1),
        block_dim=(MAMBA3_TPB, 1, 1),
    )

    # ---- working-sequence assembly (copies).
    if q0 > 0:
        ctx.enqueue_function[m3_assemble_buf_kernel](
            stages.rotq_work.unsafe_ptr(),
            stages.rotk_work.unsafe_ptr(),
            stages.v_work.unsafe_ptr(),
            stages.dt_work.unsafe_ptr(),
            stages.sig_work.unsafe_ptr(),
            stages.adt_work.unsafe_ptr(),
            state.buf_qrot.unsafe_ptr(),
            state.buf_krot.unsafe_ptr(),
            state.buf_v.unsafe_ptr(),
            state.buf_dt.unsafe_ptr(),
            state.buf_sig.unsafe_ptr(),
            state.buf_adt.unsafe_ptr(),
            Int32(b),
            Int32(l),
            Int32(q0),
            Int32(nh),
            Int32(M3_CHUNK_SIZE),
            grid_dim=(
                _grid(b * q0 * nh * (2 * n_state + p_dim + 3)),
                1,
                1,
            ),
            block_dim=(MAMBA3_TPB, 1, 1),
        )
    ctx.enqueue_function[m3_assemble_vnew_kernel](
        stages.v_work.unsafe_ptr(),
        stages.in_proj.unsafe_ptr(),
        Int32(b),
        Int32(l),
        Int32(q0),
        Int32(nh),
        Int32(dip),
        Int32(dims.col_x()),
        grid_dim=(_grid(b * l * nh * p_dim), 1, 1),
        block_dim=(MAMBA3_TPB, 1, 1),
    )
    ctx.synchronize()

    # THE REQUIRED-RED ARM (DEVIATION 831): the upstream step's own
    # per-token recurrence replaces the resumption for the new token --
    # AND IT ENGAGES ONLY AT l == 1 (the mamba2 lesson: an armed decode
    # gate runs a prefill leg through this entry point first, untouched).
    # `step_arm_engaged` is constant False in every unarmed build and the
    # branch compiles away.
    var step_arm_engaged = False
    comptime if SAB3_STEP_UPSTREAM_RECURRENCE:
        if l == 1:
            step_arm_engaged = True
            ctx.enqueue_function[m3_step_angle_kernel](
                state.theta.unsafe_ptr(),
                stages.in_proj.unsafe_ptr(),
                stages.dt_out.unsafe_ptr(),
                Int32(b),
                Int32(nh),
                Int32(dip),
                Int32(dims.col_angle()),
                grid_dim=(_grid(b * nh * r_ang), 1, 1),
                block_dim=(MAMBA3_TPB, 1, 1),
            )
            ctx.synchronize()
            ctx.enqueue_function[m3_step_core_kernel](
                stages.skip_out.unsafe_ptr(),
                stages.gate_out.unsafe_ptr(),
                state.h.unsafe_ptr(),
                state.theta.unsafe_ptr(),
                state.pend_k.unsafe_ptr(),
                state.pend_v.unsafe_ptr(),
                stages.in_proj.unsafe_ptr(),
                stages.a_out.unsafe_ptr(),
                stages.dt_out.unsafe_ptr(),
                stages.bcnorm_b.unsafe_ptr(),
                stages.bcnorm_c.unsafe_ptr(),
                w.b_bias.unsafe_ptr(),
                w.c_bias.unsafe_ptr(),
                w.d_skip.unsafe_ptr(),
                Int32(b),
                Int32(nh),
                Int32(dip),
                Int32(dims.col_z()),
                Int32(dims.col_x()),
                Int32(dims.col_trap()),
                grid_dim=(_grid(b * nh * p_dim), 1, 1),
                block_dim=(MAMBA3_TPB, 1, 1),
            )
            ctx.synchronize()
            ctx.enqueue_function[m3_step_state_kernel](
                state.pend_k.unsafe_ptr(),
                state.pend_v.unsafe_ptr(),
                state.theta.unsafe_ptr(),
                stages.in_proj.unsafe_ptr(),
                stages.bcnorm_b.unsafe_ptr(),
                w.b_bias.unsafe_ptr(),
                Int32(b),
                Int32(nh),
                Int32(dip),
                Int32(dims.col_x()),
                grid_dim=(_grid(b * nh * (n_state + p_dim)), 1, 1),
                block_dim=(MAMBA3_TPB, 1, 1),
            )
            ctx.synchronize()
            # The card's core stages are not produced by this spelling;
            # they are recorded as they stand (zeros) so the tag list
            # stays section 7's -- the gate must FAIL from the moved
            # stages onward.

    if not step_arm_engaged:
        # ---- S7-S20 + S22 + reports: the SISO core over the working
        #      sequence.
        m3_siso_forward(
            ctx,
            stages.adt_work,
            stages.sig_work,
            stages.dt_work,
            stages.gamma_work,
            stages.betap_work,
            stages.scale_work,
            stages.theta_out,
            stages.theta_last,
            stages.rotq_work,
            stages.rotk_work,
            stages.qkdot,
            stages.kscale_work,
            stages.v_work,
            stages.dacs,
            stages.seg_l,
            stages.qk_s,
            stages.pass_states,
            stages.yintra,
            stages.ystate,
            stages.skip_out,
            stages.gate_out,
            stages.h_last,
            stages.k_last,
            stages.v_last,
            stages.a_out,
            stages.dt_out,
            stages.in_proj,
            stages.bcnorm_b,
            stages.bcnorm_c,
            w.b_bias,
            w.c_bias,
            w.d_skip,
            state.theta,
            state.h,
            state.pend_k,
            state.pend_v,
            state.pending,
            b,
            l,
            q0,
            nh,
            dip,
            dims.col_z(),
            dims.col_trap(),
            dims.col_angle(),
        )
        if state.pending:
            # S22 consumed the continuation (the core applied it before
            # the state pass); the pending pieces are spent.
            state.pending = False

        # ---- the buffer update (DEVIATION 832(i)): keep the LAST
        #      WORKING CHUNK's r rows, r in [1, Q]. NOT run on an ENGAGED
        #      step arm, whose spelling carries no buffer.
        var r = t_work - (nc - 1) * qv
        ctx.enqueue_function[m3_buffer_update_kernel](
            state.buf_qrot.unsafe_ptr(),
            state.buf_krot.unsafe_ptr(),
            state.buf_v.unsafe_ptr(),
            state.buf_dt.unsafe_ptr(),
            state.buf_sig.unsafe_ptr(),
            state.buf_adt.unsafe_ptr(),
            stages.rotq_work.unsafe_ptr(),
            stages.rotk_work.unsafe_ptr(),
            stages.v_work.unsafe_ptr(),
            stages.dt_work.unsafe_ptr(),
            stages.sig_work.unsafe_ptr(),
            stages.adt_work.unsafe_ptr(),
            Int32(b),
            Int32(t_work),
            Int32(r),
            Int32(nh),
            Int32(M3_CHUNK_SIZE),
            grid_dim=(
                _grid(b * r * nh * (2 * n_state + p_dim + 3)),
                1,
                1,
            ),
            block_dim=(MAMBA3_TPB, 1, 1),
        )
        ctx.synchronize()
        state.buf_len = r

    # ---- S4: out_proj (mamba3.py:277), gemm v1 OP_NT, k = d_inner. The
    #      gate output IS the [M, d_inner] row (d = h*P + p, a copy).
    identical_gemm(
        ctx, stages.out_proj, stages.gate_out, w.w_out, m, dm, di, OP_NT
    )

    # ---- S23: residual (block.py:52/:67), the REUSED Mamba-1 kernel.
    ctx.enqueue_function[residual_add_kernel](
        stages.residual_out.unsafe_ptr(),
        x.unsafe_ptr(),
        stages.out_proj.unsafe_ptr(),
        Int32(m * dm),
        grid_dim=(_grid(m * dm), 1, 1),
        block_dim=(MAMBA3_TPB, 1, 1),
    )
    ctx.synchronize()

    # ---- the card, contract section 7's order (input.x recorded above).
    trace.record_device[DType.float32](
        ctx, prefix + ".norm.sumsq", stages.norm_sumsq, m
    )
    trace.record_device[DType.float32](
        ctx, prefix + ".norm.out", stages.norm_out, m * dm
    )
    trace.record_device[DType.float32](
        ctx, prefix + ".in_proj.out", stages.in_proj, m * dip
    )
    trace.record_device[DType.float32](
        ctx, prefix + ".A.out", stages.a_out, m * nh
    )
    trace.record_device[DType.float32](
        ctx, prefix + ".dt.out", stages.dt_out, m * nh
    )
    _record_work_slice(
        ctx, trace, prefix + ".adt.out", stages.adt_work, b, l, q0, nh
    )
    _record_work_slice(
        ctx, trace, prefix + ".trap.sigma", stages.sig_work, b, l, q0, nh
    )
    _record_work_slice(
        ctx, trace, prefix + ".trap.scale", stages.scale_work, b, l, q0, nh
    )
    trace.record_device[DType.float32](
        ctx, prefix + ".bcnorm.B", stages.bcnorm_b, m * n_state
    )
    trace.record_device[DType.float32](
        ctx, prefix + ".bcnorm.C", stages.bcnorm_c, m * n_state
    )
    trace.record_device[DType.float32](
        ctx, prefix + ".angle.theta", stages.theta_out, m * nh * r_ang
    )
    _record_work_slice(
        ctx,
        trace,
        prefix + ".rot.q",
        stages.rotq_work,
        b,
        l,
        q0,
        nh * n_state,
    )
    _record_work_slice(
        ctx,
        trace,
        prefix + ".rot.k",
        stages.rotk_work,
        b,
        l,
        q0,
        nh * n_state,
    )
    trace.record_device[DType.float32](
        ctx, prefix + ".qkdot.out", stages.qkdot, m * nh
    )
    _record_work_slice(
        ctx,
        trace,
        prefix + ".kscale.out",
        stages.kscale_work,
        b,
        l,
        q0,
        nh * n_state,
    )
    trace.record_device[DType.float32](
        ctx, prefix + ".dacs.out", stages.dacs, b * nh * nc * qv
    )
    trace.record_device[DType.float32](
        ctx, prefix + ".seg.L", stages.seg_l, b * nc * nh * qv * qv
    )
    trace.record_device[DType.float32](
        ctx, prefix + ".yintra.out", stages.yintra, m * nh * p_dim
    )
    trace.record_device[DType.float32](
        ctx, prefix + ".ystate.out", stages.ystate, m * nh * p_dim
    )
    trace.record_device[DType.float32](
        ctx, prefix + ".skip.out", stages.skip_out, m * nh * p_dim
    )
    trace.record_device[DType.float32](
        ctx, prefix + ".gate.out", stages.gate_out, m * nh * p_dim
    )
    trace.record_device[DType.float32](
        ctx, prefix + ".out_proj.out", stages.out_proj, m * dm
    )
    trace.record_device[DType.float32](
        ctx, prefix + ".residual.out", stages.residual_out, m * dm
    )
    trace.record_device[DType.float32](
        ctx,
        prefix + ".ssd.h_last",
        stages.h_last,
        b * nh * p_dim * n_state,
    )
    trace.record_device[DType.float32](
        ctx, prefix + ".ssd.k_last", stages.k_last, b * nh * n_state
    )
    trace.record_device[DType.float32](
        ctx, prefix + ".ssd.v_last", stages.v_last, b * nh * p_dim
    )
    trace.record_device[DType.float32](
        ctx, prefix + ".ssd.theta_last", stages.theta_last, b * nh * r_ang
    )


def mamba3_step(
    ctx: DeviceContext,
    mut stages: Mamba3DeviceStages,
    mut state: Mamba3DeviceState,
    mut w: Mamba3DeviceWeights,
    mut x_token: DeviceBuffer[DType.float32],
    b: Int,
    mut trace: IdentityTrace,
    prefix: String,
) raises:
    """`Mamba3.step`'s semantics (one token at a time), the profile's
    spelling: `mamba3_block_forward` at l = 1 with the state carried --
    prefill resumption, DEVIATIONS 831/832. No arithmetic of its own."""
    mamba3_block_forward(ctx, stages, state, w, x_token, b, 1, trace, prefix)
