# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`mamba_ssm/modules/mamba2.py::Mamba2.forward`, the non-mem-eff arm
(:209-276), plus `::step` (:278-343) and `::allocate_inference_cache`
(:345-355), state-spaces/mamba `e9594ce`; block order
`Mamba2Block.forward` (transformers d56c55b modeling_mamba2.py:617-631) --
ON THE DEVICE, under profile `mojolearn.identical.mamba2.fp32.v1`
(`mamba/IDENTICAL_MAMBA2_CONTRACT.md`, commit e3b46e95). **COPY, DO NOT
IMPROVE.** Seams S1-S9 and S20-S22, composed around the SSD core
(`mamba/impl/mamba_ssm/modules/ssd_minimal.mojo`, S10-S19 + h_last).

THE ENTRY POINT is `mamba2_block_forward`. Prefill is a fresh zero
`Mamba2DeviceState`; the decode step is the SAME function at `l == 1`
carrying the three-piece state (DEVIATION 786: decode is PREFILL
RESUMPTION -- the call rebuilds the open chunk from the buffer and runs
the chunked arithmetic; the carried state is the conv window, the
boundary h and the intra-chunk buffer, contract section 5). ONE spelling
for both paths is what makes gate (d) a theorem the gate verifies. The
upstream per-token recurrence (mamba2.py:310-322 /
`selective_state_update_ref`:277-282) rounds differently BY CONSTRUCTION
and is kept in this file ONLY as the required-RED arm
STEP_UPSTREAM_RECURRENCE.

INHERITED MACHINERY, REUSED, NOT RESPELLED (the contract's inheritance
column): S1-S3 and S21's fold/rstd/products run through the Mamba-1
device kernel `modeling_mamba.mojo::mamba_rms_norm` VERBATIM (same
function, same eps bits -- one spelling in the tree); S5 through its
`mamba_a_from_a_log_kernel`; the residual through its
`residual_add_kernel`; upload/download/zeros and the refusal walk through
its helpers; every projection through `gemm_identical.mojo::
identical_gemm` (OP_NT in `gemm_oracle`'s numbering -- the orientation
trap is that file's header note, honored here by importing OP_NT from
gemm_oracle and nothing else). The CONV kernel (S6) is a NEW TRANSCRIPTION
of the inherited spelling (bias-SEEDED accumulator, taps ascending,
window read -- mamba1 S13 + DEVIATION 721 verbatim) because the Mamba-1
kernel hard-addresses its `[M, 2*d_inner]` in_proj layout and this
block's xBC sits at column offset d_inner of a `[M, d_in_proj]` buffer;
only the ADDRESSING differs, the arithmetic is character for character
the same chain, and editing the Mamba-1 file to parameterize it is not
this lane's to do.

DEVIATIONS: 786 (this file's decode), 787 (gated RMSNorm gate-first,
reusing the S1-S3 machinery), 788 (`identical_clamp` at S9), 789's block
half (dt binds at S10 inside the core; this file hands the core
post-clamp dt), 790 (working-sequence stage shapes; the card records the
new-token slice of every token-shaped SSD stage -- a copy). The refusal
walk caches the weight names after the first call, DEVIATION 1886's
pattern, for DEVIATION 1886's reasons.

THE SABOTAGE ARMS THIS FILE OWNS (contract 8f):

    MOJOLEARN_MAMBA2_SABOTAGE_S6_BIAS_LAST         taps from +0.0, bias added last
    MOJOLEARN_MAMBA2_SABOTAGE_S6_TAPS_REVERSED     taps k descending
    MOJOLEARN_MAMBA2_SABOTAGE_CLAMP_BEFORE_SOFTPLUS  S9 order swapped
    MOJOLEARN_MAMBA2_SABOTAGE_GATE_NORM_BEFORE     norm_before_gate=True spelling
    MOJOLEARN_MAMBA2_SABOTAGE_STEP_UPSTREAM_RECURRENCE  mamba2.py:310-322 in decode

EVERYTHING HERE IS RUN OWED; no kernel has compiled or run. Commands live
in `mamba/checks/mamba2_check.mojo`.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace
from gemm.checks.gemm_identical import identical_gemm

# ORIENTATION NUMBERING: gemm_oracle's OP_NT = 1 (OP_NN = 0, OP_TN = 2),
# the numbering identical_gemm reads -- NOT bench/gemm_shapes.mojo's
# OP_NT = 0. The Mamba-1 header's trap note applies verbatim.
from gemm.checks.gemm_oracle import OP_NT

from checks.numerics import (
    ftz,
    identical_clamp,
    identical_exp,
    identical_mul_add,
    identical_silu,
    identical_softplus,
)
from mamba.checks.mamba2_fixture import (
    M2_CHUNK_SIZE,
    M2_D_CONV,
    M2_D_STATE,
    M2_HEADDIM,
    Mamba2Dims,
    Mamba2Weights,
)
from mamba.impl.mamba_ssm.modules.ssd_minimal import (
    MAMBA2_TPB,
    SSD_ANY_SABOTAGE,
    m2_n_chunks,
    m2_q_eff,
    ssd_forward,
    ssd_sabotage_name,
)
from mamba.impl.transformers.models.mamba.modeling_mamba import (
    _refuse_nonfinite_named,
    mamba_a_from_a_log_kernel,
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
    var g = (n + MAMBA2_TPB - 1) // MAMBA2_TPB
    if g < 1:
        return 1
    return g


# ===========================================================================
# Sabotage arms (see header).
# ===========================================================================

comptime SAB_S6_BIAS_LAST = is_defined[
    "MOJOLEARN_MAMBA2_SABOTAGE_S6_BIAS_LAST"
]()
comptime SAB_S6_TAPS_REVERSED = is_defined[
    "MOJOLEARN_MAMBA2_SABOTAGE_S6_TAPS_REVERSED"
]()
comptime SAB_CLAMP_BEFORE_SOFTPLUS = is_defined[
    "MOJOLEARN_MAMBA2_SABOTAGE_CLAMP_BEFORE_SOFTPLUS"
]()
comptime SAB_GATE_NORM_BEFORE = is_defined[
    "MOJOLEARN_MAMBA2_SABOTAGE_GATE_NORM_BEFORE"
]()
comptime SAB_STEP_UPSTREAM_RECURRENCE = is_defined[
    "MOJOLEARN_MAMBA2_SABOTAGE_STEP_UPSTREAM_RECURRENCE"
]()

comptime BLOCK2_ANY_SABOTAGE = (
    SAB_S6_BIAS_LAST
    or SAB_S6_TAPS_REVERSED
    or SAB_CLAMP_BEFORE_SOFTPLUS
    or SAB_GATE_NORM_BEFORE
    or SAB_STEP_UPSTREAM_RECURRENCE
    or SSD_ANY_SABOTAGE
)


def mamba2_sabotage_name() -> String:
    comptime if SAB_S6_BIAS_LAST:
        return String("S6_BIAS_LAST")
    comptime if SAB_S6_TAPS_REVERSED:
        return String("S6_TAPS_REVERSED")
    comptime if SAB_CLAMP_BEFORE_SOFTPLUS:
        return String("CLAMP_BEFORE_SOFTPLUS")
    comptime if SAB_GATE_NORM_BEFORE:
        return String("GATE_NORM_BEFORE")
    comptime if SAB_STEP_UPSTREAM_RECURRENCE:
        return String("STEP_UPSTREAM_RECURRENCE")
    return ssd_sabotage_name()


# ===========================================================================
# Device-side parameters, state and stages
# ===========================================================================


struct Mamba2DeviceWeights(Movable):
    """One block's parameters on the device, upstream shapes
    (`Mamba2Weights`'s table), row-major, contiguous. `weights_checked` is
    DEVIATION 1886's cache: the ten weight names are refusal-walked once,
    x and the state on every call."""

    var dims: Mamba2Dims
    var weights_checked: Bool
    var norm_w: DeviceBuffer[DType.float32]  # [d_model]
    var w_in: DeviceBuffer[DType.float32]  # [d_in_proj, d_model]
    var conv_w: DeviceBuffer[DType.float32]  # [conv_dim, 4]
    var conv_b: DeviceBuffer[DType.float32]  # [conv_dim]
    var dt_bias: DeviceBuffer[DType.float32]  # [H]
    var a_log: DeviceBuffer[DType.float32]  # [H]
    var d_skip: DeviceBuffer[DType.float32]  # [H]
    var gnorm_w: DeviceBuffer[DType.float32]  # [d_inner]
    var w_out: DeviceBuffer[DType.float32]  # [d_model, d_inner]

    def __init__(out self, ctx: DeviceContext, w: Mamba2Weights) raises:
        self.dims = w.dims.copy()
        self.weights_checked = False
        self.norm_w = mamba_upload(ctx, w.norm_w)
        self.w_in = mamba_upload(ctx, w.w_in)
        self.conv_w = mamba_upload(ctx, w.conv_w)
        self.conv_b = mamba_upload(ctx, w.conv_b)
        self.dt_bias = mamba_upload(ctx, w.dt_bias)
        self.a_log = mamba_upload(ctx, w.a_log)
        self.d_skip = mamba_upload(ctx, w.d_skip)
        self.gnorm_w = mamba_upload(ctx, w.gnorm_w)
        self.w_out = mamba_upload(ctx, w.w_out)


struct Mamba2DeviceState(Movable):
    """Contract section 5's THREE-piece state on the device (the oracle's
    `Mamba2State`, same layouts): conv window [B, CD, 4] (PRE-conv xBC,
    oldest first), boundary h [B, H, P, N], and the open chunk's buffer --
    `buf_xbc` [B, Q, CD] post-conv/post-SiLU rows and `buf_dtraw`
    [B, Q, H] raw dt rows, `buf_len` of them valid. Zeros before the
    first token (`allocate_inference_cache`, mamba2.py:345-355)."""

    var b: Int
    var dims: Mamba2Dims
    var buf_len: Int
    var conv_win: DeviceBuffer[DType.float32]
    var h: DeviceBuffer[DType.float32]
    var buf_xbc: DeviceBuffer[DType.float32]
    var buf_dtraw: DeviceBuffer[DType.float32]

    def __init__(
        out self, ctx: DeviceContext, b: Int, dims: Mamba2Dims
    ) raises:
        self.b = b
        self.dims = dims.copy()
        self.buf_len = 0
        var cd = dims.conv_dim()
        self.conv_win = mamba_zeros(ctx, b * cd * M2_D_CONV)
        self.h = mamba_zeros(
            ctx, b * dims.nheads * M2_HEADDIM * M2_D_STATE
        )
        self.buf_xbc = mamba_zeros(ctx, b * M2_CHUNK_SIZE * cd)
        self.buf_dtraw = mamba_zeros(ctx, b * M2_CHUNK_SIZE * dims.nheads)


def allocate_inference_cache(
    ctx: DeviceContext, batch_size: Int, dims: Mamba2Dims
) raises -> Mamba2DeviceState:
    """Their :345-355: ZERO state, `max_seqlen`/dtype/device dropped for
    mamba1 DEVIATION 734's reasons; the third piece (the buffer) is
    DEVIATION 786's addition and is zeros too."""
    return Mamba2DeviceState(ctx, batch_size, dims)


struct Mamba2DeviceStages(Movable):
    """Every recorded stage of one block call plus the SSD working
    buffers. Token stages are [M = B*l, ...] over the NEW tokens; SSD
    token-shaped buffers are WORKING-shaped [B, T = q0+l, ...] and the
    card records their new-token slice (DEVIATION 790). Chunk-shaped
    stages cover the working chunks at `m2_q_eff()`."""

    var b: Int
    var l: Int
    var q0: Int
    var t_work: Int
    var nc: Int
    var dims: Mamba2Dims
    var norm_sumsq: DeviceBuffer[DType.float32]  # [M]
    var norm_out: DeviceBuffer[DType.float32]  # [M, dm]
    var in_proj: DeviceBuffer[DType.float32]  # [M, dip]
    var a_out: DeviceBuffer[DType.float32]  # [H]
    var conv_out: DeviceBuffer[DType.float32]  # [M, CD]
    var silu_out: DeviceBuffer[DType.float32]  # [M, CD]
    var conv_win: DeviceBuffer[DType.float32]  # [B, CD, 4]
    var dtraw_work: DeviceBuffer[DType.float32]  # [B, T, H] raw
    var dt_work: DeviceBuffer[DType.float32]  # [B, T, H] post-S9
    var xbc_work: DeviceBuffer[DType.float32]  # [B, T, CD] post-SiLU
    var xd_work: DeviceBuffer[DType.float32]  # [B, T, H, P]
    var da_work: DeviceBuffer[DType.float32]  # [B, T, H]
    var dacs: DeviceBuffer[DType.float32]  # [B, H, C, Q]
    var seg_l: DeviceBuffer[DType.float32]  # [B, C, H, Q, Q]
    var cb_g: DeviceBuffer[DType.float32]  # [B, C, Q, Q]
    var ydiag_work: DeviceBuffer[DType.float32]  # [B, T, H, P]
    var decay: DeviceBuffer[DType.float32]  # [B, H, C, Q]
    var cstate: DeviceBuffer[DType.float32]  # [B, C, H, P, N]
    var pass_states: DeviceBuffer[DType.float32]  # [B, C, H, P, N]
    var yoff_work: DeviceBuffer[DType.float32]  # [B, T, H, P]
    var y_work: DeviceBuffer[DType.float32]  # [B, T, H, P]
    var h_last: DeviceBuffer[DType.float32]  # [B, H, P, N]
    var skip_out: DeviceBuffer[DType.float32]  # [M, H, P]
    var gnorm_gate: DeviceBuffer[DType.float32]  # [M, d_inner]
    var gnorm_sumsq: DeviceBuffer[DType.float32]  # [M]
    var gnorm_out: DeviceBuffer[DType.float32]  # [M, d_inner]
    var out_proj: DeviceBuffer[DType.float32]  # [M, dm]
    var residual_out: DeviceBuffer[DType.float32]  # [M, dm]

    def __init__(
        out self,
        ctx: DeviceContext,
        b: Int,
        l: Int,
        q0: Int,
        dims: Mamba2Dims,
    ) raises:
        self.b = b
        self.l = l
        self.q0 = q0
        self.t_work = q0 + l
        self.nc = m2_n_chunks(self.t_work)
        self.dims = dims.copy()
        var m = b * l
        var t = self.t_work
        var nc = self.nc
        var qv = m2_q_eff()
        var dm = dims.d_model
        var di = dims.d_inner
        var cd = dims.conv_dim()
        var dip = dims.d_in_proj()
        var nh = dims.nheads
        comptime p_dim = M2_HEADDIM
        comptime n_state = M2_D_STATE
        self.norm_sumsq = mamba_zeros(ctx, m)
        self.norm_out = mamba_zeros(ctx, m * dm)
        self.in_proj = mamba_zeros(ctx, m * dip)
        self.a_out = mamba_zeros(ctx, nh)
        self.conv_out = mamba_zeros(ctx, m * cd)
        self.silu_out = mamba_zeros(ctx, m * cd)
        self.conv_win = mamba_zeros(ctx, b * cd * M2_D_CONV)
        self.dtraw_work = mamba_zeros(ctx, b * t * nh)
        self.dt_work = mamba_zeros(ctx, b * t * nh)
        self.xbc_work = mamba_zeros(ctx, b * t * cd)
        self.xd_work = mamba_zeros(ctx, b * t * nh * p_dim)
        self.da_work = mamba_zeros(ctx, b * t * nh)
        self.dacs = mamba_zeros(ctx, b * nh * nc * qv)
        self.seg_l = mamba_zeros(ctx, b * nc * nh * qv * qv)
        self.cb_g = mamba_zeros(ctx, b * nc * qv * qv)
        self.ydiag_work = mamba_zeros(ctx, b * t * nh * p_dim)
        self.decay = mamba_zeros(ctx, b * nh * nc * qv)
        self.cstate = mamba_zeros(ctx, b * nc * nh * p_dim * n_state)
        self.pass_states = mamba_zeros(ctx, b * nc * nh * p_dim * n_state)
        self.yoff_work = mamba_zeros(ctx, b * t * nh * p_dim)
        self.y_work = mamba_zeros(ctx, b * t * nh * p_dim)
        self.h_last = mamba_zeros(ctx, b * nh * p_dim * n_state)
        self.skip_out = mamba_zeros(ctx, m * nh * p_dim)
        self.gnorm_gate = mamba_zeros(ctx, m * di)
        self.gnorm_sumsq = mamba_zeros(ctx, m)
        self.gnorm_out = mamba_zeros(ctx, m * di)
        self.out_proj = mamba_zeros(ctx, m * dm)
        self.residual_out = mamba_zeros(ctx, m * dm)


# ===========================================================================
# S6/S7 -- the causal depthwise conv over xBC + SiLU (mamba2.py:230-242).
# The INHERITED spelling (mamba1 S13 + DEVIATION 721): bias-SEEDED
# accumulator, taps k = 0..3 ascending (oldest first), one fma per tap,
# pre-sequence positions read the WINDOW. One thread per (batch, channel)
# walking l ascending. Only the ADDRESSING differs from the Mamba-1
# kernel: xBC sits at column offset d_inner of [M, d_in_proj].
# ===========================================================================


def m2_conv_kernel(
    conv_out: MutPointer[Float32, MutAnyOrigin],  # [M, CD]
    silu_out: MutPointer[Float32, MutAnyOrigin],  # [M, CD]
    in_proj: MutPointer[Float32, MutAnyOrigin],  # [M, dip]
    conv_w: MutPointer[Float32, MutAnyOrigin],  # [CD, 4]
    conv_b: MutPointer[Float32, MutAnyOrigin],  # [CD]
    win: MutPointer[Float32, MutAnyOrigin],  # [B, CD, 4]
    b_in: Int32,
    l_in: Int32,
    di_in: Int32,
    cd_in: Int32,
    dip_in: Int32,
):
    var b = Int(b_in)
    var l = Int(l_in)
    var di = Int(di_in)
    var cd = Int(cd_in)
    var dip = Int(dip_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= b * cd:
        return
    var bb = cell // cd
    var d = cell - bb * cd
    for li in range(l):
        var acc = ftz(conv_b.unsafe_load(d))
        comptime if SAB_S6_BIAS_LAST:
            acc = Float32(0.0)
        for kk in range(M2_D_CONV):
            var k = kk
            comptime if SAB_S6_TAPS_REVERSED:
                k = M2_D_CONV - 1 - kk
            var p = li - (M2_D_CONV - 1) + k
            var xv: Float32
            if p >= 0:
                xv = in_proj.unsafe_load((bb * l + p) * dip + di + d)
            else:
                xv = win.unsafe_load(
                    (bb * cd + d) * M2_D_CONV + (M2_D_CONV + p)
                )
            acc = ftz(
                identical_mul_add(
                    ftz(conv_w.unsafe_load(d * M2_D_CONV + k)), ftz(xv), acc
                )
            )
        comptime if SAB_S6_BIAS_LAST:
            acc = ftz(acc + ftz(conv_b.unsafe_load(d)))
        conv_out.unsafe_store((bb * l + li) * cd + d, acc)
        silu_out.unsafe_store(
            (bb * l + li) * cd + d, ftz(identical_silu(acc))
        )


def m2_conv_window_kernel(
    new_win: MutPointer[Float32, MutAnyOrigin],  # [B, CD, 4]
    in_proj: MutPointer[Float32, MutAnyOrigin],  # [M, dip]
    old_win: MutPointer[Float32, MutAnyOrigin],  # [B, CD, 4]
    b_in: Int32,
    l_in: Int32,
    di_in: Int32,
    cd_in: Int32,
    dip_in: Int32,
):
    """The window AFTER the call: the last d_conv PRE-conv xBC inputs
    (copies; out of place, mamba1 DEVIATION 726's reasons)."""
    var b = Int(b_in)
    var l = Int(l_in)
    var di = Int(di_in)
    var cd = Int(cd_in)
    var dip = Int(dip_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= b * cd * M2_D_CONV:
        return
    var bb = cell // (cd * M2_D_CONV)
    var rem = cell - bb * cd * M2_D_CONV
    var d = rem // M2_D_CONV
    var j = rem - d * M2_D_CONV
    var p = l - M2_D_CONV + j
    if p >= 0:
        new_win.unsafe_store(
            cell, in_proj.unsafe_load((bb * l + p) * dip + di + d)
        )
    else:
        new_win.unsafe_store(
            cell,
            old_win.unsafe_load((bb * cd + d) * M2_D_CONV + (M2_D_CONV + p)),
        )


# ===========================================================================
# Working-sequence assembly (copies, not seams): buffered rows ++ new rows.
# ===========================================================================


def m2_assemble_xbc_kernel(
    xbc_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, CD]
    buf_xbc: MutPointer[Float32, MutAnyOrigin],  # [B, Q, CD]
    silu_out: MutPointer[Float32, MutAnyOrigin],  # [M, CD]
    b_in: Int32,
    l_in: Int32,
    q0_in: Int32,
    cd_in: Int32,
    q_cap_in: Int32,
):
    var b = Int(b_in)
    var l = Int(l_in)
    var q0 = Int(q0_in)
    var cd = Int(cd_in)
    var q_cap = Int(q_cap_in)
    var t_work = q0 + l
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= b * t_work * cd:
        return
    var bb = cell // (t_work * cd)
    var rem = cell - bb * t_work * cd
    var t = rem // cd
    var d = rem - t * cd
    if t < q0:
        xbc_work.unsafe_store(cell, buf_xbc.unsafe_load((bb * q_cap + t) * cd + d))
    else:
        xbc_work.unsafe_store(
            cell, silu_out.unsafe_load((bb * l + (t - q0)) * cd + d)
        )


def m2_assemble_dtraw_kernel(
    dtraw_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, H]
    buf_dtraw: MutPointer[Float32, MutAnyOrigin],  # [B, Q, H]
    in_proj: MutPointer[Float32, MutAnyOrigin],  # [M, dip]
    b_in: Int32,
    l_in: Int32,
    q0_in: Int32,
    nh_in: Int32,
    di_in: Int32,
    cd_in: Int32,
    dip_in: Int32,
    q_cap_in: Int32,
):
    var b = Int(b_in)
    var l = Int(l_in)
    var q0 = Int(q0_in)
    var nh = Int(nh_in)
    var di = Int(di_in)
    var cd = Int(cd_in)
    var dip = Int(dip_in)
    var q_cap = Int(q_cap_in)
    var t_work = q0 + l
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= b * t_work * nh:
        return
    var bb = cell // (t_work * nh)
    var rem = cell - bb * t_work * nh
    var t = rem // nh
    var hh = rem - t * nh
    if t < q0:
        dtraw_work.unsafe_store(
            cell, buf_dtraw.unsafe_load((bb * q_cap + t) * nh + hh)
        )
    else:
        dtraw_work.unsafe_store(
            cell,
            in_proj.unsafe_load((bb * l + (t - q0)) * dip + di + cd + hh),
        )


# ===========================================================================
# S9 -- dt = identical_clamp(softplus(dt_raw + dt_bias), lo, hi), bias ->
# softplus -> clamp IN THAT ORDER (`_chunk_cumsum_fwd_kernel`:73-81; HF
# :272-276; DEVIATION 788). Over the WORKING rows (S9 is re-derived for
# buffered rows from their raw dt: a pure function of the same bits).
# ===========================================================================


def m2_dt_kernel(
    dt_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, H]
    dtraw_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, H]
    dt_bias: MutPointer[Float32, MutAnyOrigin],  # [H]
    n_in: Int32,  # B * T * H
    nh_in: Int32,
    dt_lo: Float32,
    dt_hi: Float32,
):
    var n = Int(n_in)
    var nh = Int(nh_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= n:
        return
    var hh = cell - (cell // nh) * nh
    var biased = ftz(
        ftz(dtraw_work.unsafe_load(cell)) + ftz(dt_bias.unsafe_load(hh))
    )
    comptime if SAB_CLAMP_BEFORE_SOFTPLUS:
        # SABOTAGE: the S9 order swapped -- clamp the biased value, then
        # softplus. Witnessed only by an ACTIVE dt_limit fixture.
        var clamped = ftz(identical_clamp(biased, dt_lo, dt_hi))
        dt_work.unsafe_store(cell, ftz(identical_softplus(clamped)))
    else:
        var sp = ftz(identical_softplus(biased))
        dt_work.unsafe_store(cell, ftz(identical_clamp(sp, dt_lo, dt_hi)))


# ===========================================================================
# Buffer update (copies): keep the last r = T mod Q working rows.
# ===========================================================================


def m2_buffer_update_kernel(
    buf_xbc: MutPointer[Float32, MutAnyOrigin],  # [B, Q, CD]
    buf_dtraw: MutPointer[Float32, MutAnyOrigin],  # [B, Q, H]
    xbc_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, CD]
    dtraw_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, H]
    b_in: Int32,
    t_in: Int32,
    r_in: Int32,
    cd_in: Int32,
    nh_in: Int32,
    q_cap_in: Int32,
):
    var b = Int(b_in)
    var t_work = Int(t_in)
    var r = Int(r_in)
    var cd = Int(cd_in)
    var nh = Int(nh_in)
    var q_cap = Int(q_cap_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= b * r * (cd + nh):
        return
    var bb = cell // (r * (cd + nh))
    var rem = cell - bb * r * (cd + nh)
    var t = rem // (cd + nh)
    var d = rem - t * (cd + nh)
    var src = t_work - r + t
    if d < cd:
        buf_xbc.unsafe_store(
            (bb * q_cap + t) * cd + d,
            xbc_work.unsafe_load((bb * t_work + src) * cd + d),
        )
    else:
        buf_dtraw.unsafe_store(
            (bb * q_cap + t) * nh + (d - cd),
            dtraw_work.unsafe_load((bb * t_work + src) * nh + (d - cd)),
        )


# ===========================================================================
# S20 -- out = Y + x * D[h]: D-residual from UNDISCRETIZED post-conv x,
# PRODUCT + add, D LAST (HF :284-285, :338-339; mamba1 S11's decision).
# Reads the working Y at row q0 + li, writes the [M, H, P] stage.
# ===========================================================================


def m2_skip_kernel(
    skip_out: MutPointer[Float32, MutAnyOrigin],  # [M, H, P]
    y_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, H, P]
    silu_out: MutPointer[Float32, MutAnyOrigin],  # [M, CD]
    d_skip: MutPointer[Float32, MutAnyOrigin],  # [H]
    b_in: Int32,
    l_in: Int32,
    q0_in: Int32,
    nh_in: Int32,
    cd_in: Int32,
):
    var b = Int(b_in)
    var l = Int(l_in)
    var q0 = Int(q0_in)
    var nh = Int(nh_in)
    var cd = Int(cd_in)
    var t_work = q0 + l
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= b * l * nh * M2_HEADDIM:
        return
    var bb = cell // (l * nh * M2_HEADDIM)
    var rem = cell - bb * l * nh * M2_HEADDIM
    var li = rem // (nh * M2_HEADDIM)
    var rem2 = rem - li * nh * M2_HEADDIM
    var hh = rem2 // M2_HEADDIM
    var p = rem2 - hh * M2_HEADDIM
    var xv = ftz(
        silu_out.unsafe_load((bb * l + li) * cd + hh * M2_HEADDIM + p)
    )
    var prod = ftz(pinned_mul(xv, ftz(d_skip.unsafe_load(hh))))
    var y = ftz(
        y_work.unsafe_load(
            (((bb * t_work) + (q0 + li)) * nh + hh) * M2_HEADDIM + p
        )
    )
    skip_out.unsafe_store(cell, ftz(y + prod))


# ===========================================================================
# S21's gate -- y_g = y * silu(z), gate BEFORE norm (norm_before_gate =
# False, rms_norm_ref:26-27; DEVIATION 787). The fold/rstd/products after
# it are the REUSED Mamba-1 mamba_rms_norm. skip_out [M, H, P] and the
# gate row [M, d_ssm] are the same bytes (d = h*P + p).
# ===========================================================================


def m2_gate_kernel(
    gate_out: MutPointer[Float32, MutAnyOrigin],  # [M, d_inner]
    y_in: MutPointer[Float32, MutAnyOrigin],  # [M, d_inner]
    in_proj: MutPointer[Float32, MutAnyOrigin],  # [M, dip]
    n_in: Int32,  # M * d_inner
    di_in: Int32,
    dip_in: Int32,
):
    var n = Int(n_in)
    var di = Int(di_in)
    var dip = Int(dip_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= n:
        return
    var t = cell // di
    var j = cell - t * di
    var z = ftz(in_proj.unsafe_load(t * dip + j))
    gate_out.unsafe_store(
        cell,
        ftz(
            pinned_mul(
                ftz(y_in.unsafe_load(cell)), ftz(identical_silu(z))
            )
        ),
    )


# ===========================================================================
# THE REQUIRED-RED ARM: STEP_UPSTREAM_RECURRENCE (mamba2.py:310-322 /
# selective_state_update_ref:277-282). The upstream torch step's own
# rounding: dt WITHOUT the clamp, dA = exp(dt*A), dBx = (dt*B)*x pairing,
# h = h*dA + dBx (unfused mul/add as torch rounds it), y = C.h_new,
# y += D*x. One thread per (b, h, p); h updated IN PLACE. Exists to be
# falsified by gate (d); never the profile.
# ===========================================================================


def m2_step_upstream_kernel(
    skip_out: MutPointer[Float32, MutAnyOrigin],  # [M = B, H, P]
    h_state: MutPointer[Float32, MutAnyOrigin],  # [B, H, P, N]
    silu_out: MutPointer[Float32, MutAnyOrigin],  # [B, CD] (l = 1)
    in_proj: MutPointer[Float32, MutAnyOrigin],  # [B, dip]
    dt_bias: MutPointer[Float32, MutAnyOrigin],  # [H]
    a_out: MutPointer[Float32, MutAnyOrigin],  # [H]
    d_skip: MutPointer[Float32, MutAnyOrigin],  # [H]
    b_in: Int32,
    nh_in: Int32,
    di_in: Int32,
    cd_in: Int32,
    dip_in: Int32,
):
    var b = Int(b_in)
    var nh = Int(nh_in)
    var di = Int(di_in)
    var cd = Int(cd_in)
    var dip = Int(dip_in)
    comptime n_state = M2_D_STATE
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= b * nh * M2_HEADDIM:
        return
    var bb = cell // (nh * M2_HEADDIM)
    var rem = cell - bb * nh * M2_HEADDIM
    var hh = rem // M2_HEADDIM
    var p = rem - hh * M2_HEADDIM
    # dt = softplus(dt_raw + bias) -- NO clamp in their step (:313).
    var raw = ftz(in_proj.unsafe_load(bb * dip + di + cd + hh))
    var dtv = ftz(identical_softplus(ftz(raw + ftz(dt_bias.unsafe_load(hh)))))
    var dav = ftz(identical_exp(ftz(pinned_mul(dtv, ftz(a_out.unsafe_load(hh))))))
    var xv = ftz(silu_out.unsafe_load(bb * cd + hh * M2_HEADDIM + p))
    var acc = Float32(0.0)
    for n in range(n_state):
        var bv = ftz(silu_out.unsafe_load(bb * cd + di + n))
        var db = ftz(pinned_mul(dtv, bv))  # (dt * B) first (:277)
        var dbx = ftz(pinned_mul(db, xv))  # ... then * x
        var hidx = ((bb * nh + hh) * M2_HEADDIM + p) * n_state + n
        var hprev = ftz(h_state.unsafe_load(hidx))
        # h*dA + dBx, TWO roundings (torch's mul then add).
        var hnew = ftz(ftz(pinned_mul(hprev, dav)) + dbx)
        h_state.unsafe_store(hidx, hnew)
        var cv = ftz(silu_out.unsafe_load(bb * cd + di + n_state + n))
        acc = ftz(identical_mul_add(cv, hnew, acc))
    var prod = ftz(pinned_mul(xv, ftz(d_skip.unsafe_load(hh))))
    skip_out.unsafe_store(cell, ftz(ftz(acc) + prod))


# ===========================================================================
# Refusal (contract section 6): every named input, BY BITS, before any
# recorded stage. Weight names cached after the first call (DEVIATION
# 1886's pattern); x and the state walked every call.
# ===========================================================================


def mamba2_refuse_bad_inputs(
    ctx: DeviceContext,
    mut w: Mamba2DeviceWeights,
    mut x: DeviceBuffer[DType.float32],
    mut state: Mamba2DeviceState,
    b: Int,
    l: Int,
) raises:
    var dims = w.dims.copy()
    var dm = dims.d_model
    var di = dims.d_inner
    var cd = dims.conv_dim()
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
            "conv1d.weight", mamba_download(ctx, w.conv_w, cd * M2_D_CONV)
        )
        _refuse_nonfinite_named(
            "conv1d.bias", mamba_download(ctx, w.conv_b, cd)
        )
        _refuse_nonfinite_named(
            "dt_bias", mamba_download(ctx, w.dt_bias, nh)
        )
        _refuse_nonfinite_named("A_log", mamba_download(ctx, w.a_log, nh))
        _refuse_nonfinite_named("D", mamba_download(ctx, w.d_skip, nh))
        _refuse_nonfinite_named(
            "norm_gated.weight", mamba_download(ctx, w.gnorm_w, di)
        )
        _refuse_nonfinite_named(
            "out_proj.weight", mamba_download(ctx, w.w_out, dm * di)
        )
        w.weights_checked = True
    _refuse_nonfinite_named(
        "state.conv_win",
        mamba_download(ctx, state.conv_win, b * cd * M2_D_CONV),
    )
    _refuse_nonfinite_named(
        "state.h",
        mamba_download(ctx, state.h, b * nh * M2_HEADDIM * M2_D_STATE),
    )
    _refuse_nonfinite_named(
        "state.buf_xbc",
        mamba_download(ctx, state.buf_xbc, b * M2_CHUNK_SIZE * cd),
    )
    _refuse_nonfinite_named(
        "state.buf_dtraw",
        mamba_download(ctx, state.buf_dtraw, b * M2_CHUNK_SIZE * nh),
    )


# ===========================================================================
# Card recording helpers (DEVIATION 790: token-shaped SSD stages are
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


def mamba2_block_forward(
    ctx: DeviceContext,
    mut stages: Mamba2DeviceStages,
    mut state: Mamba2DeviceState,
    mut w: Mamba2DeviceWeights,
    mut x: DeviceBuffer[DType.float32],
    b: Int,
    l: Int,
    dt_lo: Float32,
    dt_hi: Float32,
    mut trace: IdentityTrace,
    prefix: String,
) raises:
    """One block call (Mamba2Block.forward around the non-mem-eff mixer),
    every stage of contract section 7 recorded under `prefix` in the
    section's order. Prefill = fresh zero state; decode = same entry,
    state carried (DEVIATION 786)."""
    if b <= 0 or l <= 0:
        raise Error(
            "mamba2_block_forward: B and L must be positive, got B="
            + String(b)
            + " L="
            + String(l)
        )
    if stages.b != b or stages.l != l or stages.q0 != state.buf_len:
        raise Error(
            "mamba2_block_forward: stages were built for B="
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
        raise Error("mamba2_block_forward: the state's shape is not the call's")
    if w.dims.d_model != stages.dims.d_model:
        raise Error(
            "mamba2_block_forward: the weights' d_model is not the stages'"
        )

    mamba2_refuse_bad_inputs(ctx, w, x, state, b, l)

    var dims = stages.dims.copy()
    var dm = dims.d_model
    var di = dims.d_inner
    var cd = dims.conv_dim()
    var dip = dims.d_in_proj()
    var nh = dims.nheads
    comptime p_dim = M2_HEADDIM
    comptime n_state = M2_D_STATE
    var m = b * l
    var q0 = state.buf_len
    var t_work = q0 + l
    var qv = m2_q_eff()
    var nc = stages.nc

    trace.record_device[DType.float32](ctx, prefix + ".input.x", x, m * dm)

    # ---- S1-S3: the block RMSNorm, the REUSED Mamba-1 kernel.
    mamba_rms_norm(
        ctx, stages.norm_sumsq, stages.norm_out, x, w.norm_w, m, dm
    )
    ctx.synchronize()
    trace.record_device[DType.float32](
        ctx, prefix + ".norm.sumsq", stages.norm_sumsq, m
    )
    trace.record_device[DType.float32](
        ctx, prefix + ".norm.out", stages.norm_out, m * dm
    )

    # ---- S4: in_proj (mamba2.py:211), gemm v1 OP_NT, k = d_model.
    identical_gemm(
        ctx, stages.in_proj, stages.norm_out, w.w_in, m, dip, dm, OP_NT
    )
    trace.record_device[DType.float32](
        ctx, prefix + ".in_proj.out", stages.in_proj, m * dip
    )

    # ---- S5: A = -exp(A_log) (mamba2.py:182), the REUSED Mamba-1 kernel.
    ctx.enqueue_function[mamba_a_from_a_log_kernel](
        stages.a_out.unsafe_ptr(),
        w.a_log.unsafe_ptr(),
        Int32(nh),
        grid_dim=(_grid(nh), 1, 1),
        block_dim=(MAMBA2_TPB, 1, 1),
    )
    ctx.synchronize()
    trace.record_device[DType.float32](
        ctx, prefix + ".A.out", stages.a_out, nh
    )

    # ---- S6/S7: conv + SiLU over xBC; window updated out of place.
    ctx.enqueue_function[m2_conv_kernel](
        stages.conv_out.unsafe_ptr(),
        stages.silu_out.unsafe_ptr(),
        stages.in_proj.unsafe_ptr(),
        w.conv_w.unsafe_ptr(),
        w.conv_b.unsafe_ptr(),
        state.conv_win.unsafe_ptr(),
        Int32(b),
        Int32(l),
        Int32(di),
        Int32(cd),
        Int32(dip),
        grid_dim=(_grid(b * cd), 1, 1),
        block_dim=(MAMBA2_TPB, 1, 1),
    )
    ctx.enqueue_function[m2_conv_window_kernel](
        stages.conv_win.unsafe_ptr(),
        stages.in_proj.unsafe_ptr(),
        state.conv_win.unsafe_ptr(),
        Int32(b),
        Int32(l),
        Int32(di),
        Int32(cd),
        Int32(dip),
        grid_dim=(_grid(b * cd * M2_D_CONV), 1, 1),
        block_dim=(MAMBA2_TPB, 1, 1),
    )
    ctx.synchronize()
    trace.record_device[DType.float32](
        ctx, prefix + ".conv.out", stages.conv_out, m * cd
    )
    trace.record_device[DType.float32](
        ctx, prefix + ".silu.out", stages.silu_out, m * cd
    )
    trace.record_device[DType.float32](
        ctx, prefix + ".conv.window", stages.conv_win, b * cd * M2_D_CONV
    )
    ctx.enqueue_copy(dst_buf=state.conv_win, src_buf=stages.conv_win)
    ctx.synchronize()

    # ---- working-sequence assembly (copies) + S9 over working rows.
    ctx.enqueue_function[m2_assemble_xbc_kernel](
        stages.xbc_work.unsafe_ptr(),
        state.buf_xbc.unsafe_ptr(),
        stages.silu_out.unsafe_ptr(),
        Int32(b),
        Int32(l),
        Int32(q0),
        Int32(cd),
        Int32(M2_CHUNK_SIZE),
        grid_dim=(_grid(b * t_work * cd), 1, 1),
        block_dim=(MAMBA2_TPB, 1, 1),
    )
    ctx.enqueue_function[m2_assemble_dtraw_kernel](
        stages.dtraw_work.unsafe_ptr(),
        state.buf_dtraw.unsafe_ptr(),
        stages.in_proj.unsafe_ptr(),
        Int32(b),
        Int32(l),
        Int32(q0),
        Int32(nh),
        Int32(di),
        Int32(cd),
        Int32(dip),
        Int32(M2_CHUNK_SIZE),
        grid_dim=(_grid(b * t_work * nh), 1, 1),
        block_dim=(MAMBA2_TPB, 1, 1),
    )
    ctx.enqueue_function[m2_dt_kernel](
        stages.dt_work.unsafe_ptr(),
        stages.dtraw_work.unsafe_ptr(),
        w.dt_bias.unsafe_ptr(),
        Int32(b * t_work * nh),
        Int32(nh),
        dt_lo,
        dt_hi,
        grid_dim=(_grid(b * t_work * nh), 1, 1),
        block_dim=(MAMBA2_TPB, 1, 1),
    )
    ctx.synchronize()
    _record_work_slice(
        ctx, trace, prefix + ".dt.out", stages.dt_work, b, l, q0, nh
    )

    comptime if SAB_STEP_UPSTREAM_RECURRENCE:
        # THE REQUIRED-RED ARM (DEVIATION 786): the upstream step's own
        # per-token recurrence replaces the resumption for the new token.
        # Only meaningful at l == 1; gate (d) must FAIL against a prefill.
        if l != 1:
            raise Error(
                "STEP_UPSTREAM_RECURRENCE is a decode arm; run it at l = 1"
            )
        ctx.enqueue_function[m2_step_upstream_kernel](
            stages.skip_out.unsafe_ptr(),
            state.h.unsafe_ptr(),
            stages.silu_out.unsafe_ptr(),
            stages.in_proj.unsafe_ptr(),
            w.dt_bias.unsafe_ptr(),
            stages.a_out.unsafe_ptr(),
            w.d_skip.unsafe_ptr(),
            Int32(b),
            Int32(nh),
            Int32(di),
            Int32(cd),
            Int32(dip),
            grid_dim=(_grid(b * nh * p_dim), 1, 1),
            block_dim=(MAMBA2_TPB, 1, 1),
        )
        ctx.synchronize()
        # The card's SSD stages are not produced by this spelling; record
        # the working buffers as they stand (zeros) so the tag list stays
        # section 7's -- the gate reads skip.out onward, which is where
        # this arm must bite.
    else:
        # ---- S10-S19 + h_last: the SSD core over the working sequence.
        ssd_forward(
            ctx,
            stages.xd_work,
            stages.da_work,
            stages.dacs,
            stages.seg_l,
            stages.cb_g,
            stages.ydiag_work,
            stages.decay,
            stages.cstate,
            stages.pass_states,
            stages.yoff_work,
            stages.y_work,
            stages.h_last,
            stages.xbc_work,
            stages.dt_work,
            stages.a_out,
            state.h,
            b,
            t_work,
            di,
            cd,
            nh,
        )

    _record_work_slice(
        ctx, trace, prefix + ".xd.out", stages.xd_work, b, l, q0, nh * p_dim
    )
    trace.record_device[DType.float32](
        ctx, prefix + ".dacs.out", stages.dacs, b * nh * nc * qv
    )
    trace.record_device[DType.float32](
        ctx, prefix + ".seg.L", stages.seg_l, b * nc * nh * qv * qv
    )
    trace.record_device[DType.float32](
        ctx, prefix + ".cb.G", stages.cb_g, b * nc * qv * qv
    )
    _record_work_slice(
        ctx,
        trace,
        prefix + ".ydiag.out",
        stages.ydiag_work,
        b,
        l,
        q0,
        nh * p_dim,
    )
    trace.record_device[DType.float32](
        ctx, prefix + ".decay.states", stages.decay, b * nh * nc * qv
    )
    trace.record_device[DType.float32](
        ctx,
        prefix + ".cstate.out",
        stages.cstate,
        b * nc * nh * p_dim * n_state,
    )
    trace.record_device[DType.float32](
        ctx,
        prefix + ".pass.states",
        stages.pass_states,
        b * nc * nh * p_dim * n_state,
    )
    _record_work_slice(
        ctx,
        trace,
        prefix + ".yoff.out",
        stages.yoff_work,
        b,
        l,
        q0,
        nh * p_dim,
    )
    _record_work_slice(
        ctx, trace, prefix + ".scan.y", stages.y_work, b, l, q0, nh * p_dim
    )

    # ---- the buffer update: keep the last r = T mod Q working rows
    #      (copies; r == 0 empties the buffer). NOT run under the step
    #      arm, whose spelling carries no buffer.
    comptime if not SAB_STEP_UPSTREAM_RECURRENCE:
        var r = t_work - (t_work // qv) * qv
        if r > 0:
            ctx.enqueue_function[m2_buffer_update_kernel](
                state.buf_xbc.unsafe_ptr(),
                state.buf_dtraw.unsafe_ptr(),
                stages.xbc_work.unsafe_ptr(),
                stages.dtraw_work.unsafe_ptr(),
                Int32(b),
                Int32(t_work),
                Int32(r),
                Int32(cd),
                Int32(nh),
                Int32(M2_CHUNK_SIZE),
                grid_dim=(_grid(b * r * (cd + nh)), 1, 1),
                block_dim=(MAMBA2_TPB, 1, 1),
            )
            ctx.synchronize()
        state.buf_len = r

    # ---- S20: the D residual (skipped by the step arm, which wrote
    #      skip_out itself).
    comptime if not SAB_STEP_UPSTREAM_RECURRENCE:
        ctx.enqueue_function[m2_skip_kernel](
            stages.skip_out.unsafe_ptr(),
            stages.y_work.unsafe_ptr(),
            stages.silu_out.unsafe_ptr(),
            w.d_skip.unsafe_ptr(),
            Int32(b),
            Int32(l),
            Int32(q0),
            Int32(nh),
            Int32(cd),
            grid_dim=(_grid(m * nh * p_dim), 1, 1),
            block_dim=(MAMBA2_TPB, 1, 1),
        )
        ctx.synchronize()
    trace.record_device[DType.float32](
        ctx, prefix + ".skip.out", stages.skip_out, m * nh * p_dim
    )

    # ---- S21: gate FIRST (DEVIATION 787), then the REUSED Mamba-1
    #      RMSNorm machinery over d_ssm = d_inner.
    comptime if SAB_GATE_NORM_BEFORE:
        # SABOTAGE: norm_before_gate=True's spelling -- norm skip_out
        # first, gate AFTER. gnorm.gate then holds the pre-gate normed
        # rows, and everything from gnorm.gate onward must move.
        mamba_rms_norm(
            ctx,
            stages.gnorm_sumsq,
            stages.gnorm_gate,
            stages.skip_out,
            w.gnorm_w,
            m,
            di,
        )
        ctx.synchronize()
        ctx.enqueue_function[m2_gate_kernel](
            stages.gnorm_out.unsafe_ptr(),
            stages.gnorm_gate.unsafe_ptr(),
            stages.in_proj.unsafe_ptr(),
            Int32(m * di),
            Int32(di),
            Int32(dip),
            grid_dim=(_grid(m * di), 1, 1),
            block_dim=(MAMBA2_TPB, 1, 1),
        )
        ctx.synchronize()
    else:
        ctx.enqueue_function[m2_gate_kernel](
            stages.gnorm_gate.unsafe_ptr(),
            stages.skip_out.unsafe_ptr(),
            stages.in_proj.unsafe_ptr(),
            Int32(m * di),
            Int32(di),
            Int32(dip),
            grid_dim=(_grid(m * di), 1, 1),
            block_dim=(MAMBA2_TPB, 1, 1),
        )
        ctx.synchronize()
        mamba_rms_norm(
            ctx,
            stages.gnorm_sumsq,
            stages.gnorm_out,
            stages.gnorm_gate,
            w.gnorm_w,
            m,
            di,
        )
        ctx.synchronize()
    trace.record_device[DType.float32](
        ctx, prefix + ".gnorm.gate", stages.gnorm_gate, m * di
    )
    trace.record_device[DType.float32](
        ctx, prefix + ".gnorm.sumsq", stages.gnorm_sumsq, m
    )
    trace.record_device[DType.float32](
        ctx, prefix + ".gnorm.out", stages.gnorm_out, m * di
    )

    # ---- S4: out_proj (mamba2.py:275), gemm v1 OP_NT, k = d_inner.
    identical_gemm(
        ctx, stages.out_proj, stages.gnorm_out, w.w_out, m, dm, di, OP_NT
    )
    trace.record_device[DType.float32](
        ctx, prefix + ".out_proj.out", stages.out_proj, m * dm
    )

    # ---- S22: residual (HF :630), the REUSED Mamba-1 kernel.
    ctx.enqueue_function[residual_add_kernel](
        stages.residual_out.unsafe_ptr(),
        x.unsafe_ptr(),
        stages.out_proj.unsafe_ptr(),
        Int32(m * dm),
        grid_dim=(_grid(m * dm), 1, 1),
        block_dim=(MAMBA2_TPB, 1, 1),
    )
    ctx.synchronize()
    trace.record_device[DType.float32](
        ctx, prefix + ".residual.out", stages.residual_out, m * dm
    )

    # ---- section 5's report stage.
    trace.record_device[DType.float32](
        ctx, prefix + ".ssd.h_last", stages.h_last, b * nh * p_dim * n_state
    )


def mamba2_step(
    ctx: DeviceContext,
    mut stages: Mamba2DeviceStages,
    mut state: Mamba2DeviceState,
    mut w: Mamba2DeviceWeights,
    mut x_token: DeviceBuffer[DType.float32],
    b: Int,
    dt_lo: Float32,
    dt_hi: Float32,
    mut trace: IdentityTrace,
    prefix: String,
) raises:
    """`Mamba2.step`'s semantics (:280 "1 token at a time"), the profile's
    spelling: `mamba2_block_forward` at l = 1 with the state carried --
    prefill resumption, DEVIATION 786. No arithmetic of its own."""
    mamba2_block_forward(
        ctx, stages, state, w, x_token, b, 1, dt_lo, dt_hi, trace, prefix
    )
