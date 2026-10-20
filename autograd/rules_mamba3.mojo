# SPDX-License-Identifier: Apache-2.0
# Mamba-3 block backward on the tape, decomposed into stage rules
# (TAPE_EVERYWHERE_PLAN.md path 1, step 1b): the device's only Mamba-3
# backward. The GPU binding (mamba3_tape_backward_on, mamba3_tape_backward)
# and the Samba train step (autograd/rules_samba.mojo, which composes these
# rules per Mamba-3 layer) record the same five stage nodes per block.
#
# ============================================================ LAUNCH INVENTORY
# Hand path (the deleted device walk mamba3_prefill_backward_on; the CPU
# column keeps it as generator input, mamba/host/mamba3_prefill_chain.mojo),
# one block, zero state, forward = mamba3_block_forward:
#   D1  proj_a PROJ3_OUT d_gate         proj_b PROJ3_OUT d_w_out
#   D2  gate_skip  d_skip, d_z, d_v, d_qkdot, d_d_product;  reduce RED3_D
#   D3  qkdot, s16_s15, join_rotary, [dead pass: join_current, angle,
#       dt_partial, reduce RED3_DT_BIAS], seg_adt, [dead: adt_product],
#       s17_state, s17_operands, join_s16_s17, s15_only, rotary_only,
#       dacs_to_adt, join_two, adt_product, a_heavy_tail, copy d_beta,
#       join_current, angle, join_two, dt_partial, reduce RED3_DT_BIAS,
#       2 x bcnorm, reduce RED3_BNORM_W / RED3_CNORM_W, RED3_B_BIAS /
#       RED3_C_BIAS, pack_in_proj -> d_in_proj
#   D4  proj_a PROJ3_IN d_norm          proj_b PROJ3_IN d_w_in
#   D5  block_norm d_x = d_res + norm'(d_norm), rows -> reduce RED3_NORM_W
#   then one synchronize (the walk's closing wait).
# ([dead] = the IDN_M3_BWD_SKIP_DEAD stages, skipped by default, run under
# -D MOJOLEARN_IDN_M3_BWD_SKIP_DEAD_OFF exactly as the walk ran them.)
#
# ============================================================ THE TAPE GRAPH
# The forward is the hand path's (mamba3_block_forward, stages recorded);
# the tape records nodes over those stage buffers and relaunches nothing.
# Per block, with x the block input and the nine weights as leaves in the
# registry order [norm_w, w_in, dt_bias, bnorm_w, cnorm_w, b_bias, c_bias,
# D, w_out]:
#   NRM = M3_NORM(x, norm_w)                          st.norm_out   D5
#   P   = M3_PROJ_IN(NRM, w_in)                       st.in_proj    D4
#   SK  = M3_SSD(P, dt_bias, bnorm_w, cnorm_w, b_bias, c_bias)
#                                                     st.skip_out   D3
#   G   = M3_GATE(SK, P, D)                           st.gate_out   D2
#   O   = M3_PROJ_OUT(G, w_out)                       st.out_proj   D1
#   y   = M3_RESID(x, O)                              st.residual_out
# The tape runs them in reverse record order: D1, D2, D3, D4, D5, the hand
# path's order, with the hand path's arguments on the hand path's buffers.
#   * Residual. M3_RESID launches nothing: O's gradient storage is the
#     block output's (custom_into_shared), the caller's d_output. x's whole
#     gradient is written by M3_NORM (writes_total on both x edges), whose
#     block_norm kernel already adds the residual term (saved.p2).
#   * P's two consumers. M3_GATE leaves d_z and d_v in the block's scratch
#     (its P edge is writes_total and untouched); M3_SSD, P's last consumer
#     in reverse order, packs P's whole gradient (D3's pack_in_proj).
#   * The qkdot terms D2 hands D3 (d_qkdot) and every other intermediate the
#     walk allocates live in the block's Mamba3TapeScratch, allocated zeroed
#     at the block's first rule (M3_PROJ_OUT) as the walk allocates them, and
#     freed after the walk's closing synchronize at its last (M3_NORM), so a
#     stack holds one block's scratch at a time, as the walk did.
#   * zeroed_destination is False everywhere: every destination is stored
#     whole by a GEMM, a reduce, the block norm or the pack.
# Expected result: the hand path's bits on every gradient word (the same
# kernels on the same operands in the same order). No new kernel, no opcode.
#
# ============================================================ LAUNCH COUNT
# Per block: the hand launches D1-D5 exactly, no zero fill or accumulate of
# the tape's, and no copy: the ten gradients are written in place (the old
# opaque M3_BLOCK rule copied each of them once more). One wait per block,
# as before (D5's closing wait), then the scratch goes.
#
# Import direction: this module imports the mamba modules; no mamba module
# imports autograd.
from std.os import getenv
from std.sys.compile import is_defined
from std.time import perf_counter_ns
from max.gpu.host import DeviceBuffer, DeviceContext

from autograd.tape import BackwardRules, Tape, Tensor
from sequence.exec_trait import Exec
from sequence.exec_device import DeviceExec
from sequence.ops import FP, Args

from core.identity_trace import IdentityTrace
from mamba.checks.mamba3_backward import (
    PROJ3_IN, PROJ3_OUT,
    RED3_D, RED3_NORM_W,
    RED3_DT_BIAS,
    RED3_BNORM_W, RED3_CNORM_W, RED3_B_BIAS, RED3_C_BIAS,
    mamba3_backward_ones_floats,
    mamba3_backward_proj_a_into,
    mamba3_backward_proj_b_into,
    mamba3_backward_reduce_into,
    mamba3_backward_workspace_max_floats,
)
from mamba.checks.mamba3_fixture import (
    M3_CHUNK_SIZE,
    M3_D_STATE,
    M3_HEADDIM,
    M3_NUM_ROPE_ANGLES,
    Mamba3Dims,
    Mamba3Weights,
)
from mamba.impl.modules.mamba3_refusal import mamba_refuse_grad_output
from mamba.impl.modules.mamba3 import (
    Mamba3DeviceStages,
    Mamba3DeviceWeights,
    allocate_inference_cache,
    mamba3_block_forward,
)
from mamba.impl.modules.mamba3_backward import (
    AFN_M3_BWD_CHUNK,
    afn_m3_theta_chunks,
    mamba3_afn_backward_angle_into,
    mamba3_backward_gate_skip_into,
    mamba3_backward_qkdot_into,
    mamba3_backward_s16_s15_into,
    mamba3_backward_join_rotary_into,
    mamba3_backward_join_current_into,
    mamba3_backward_angle_into,
    mamba3_angle_suffix_sums_cells,
    mamba3_backward_dt_partial_into,
    mamba3_backward_seg_adt_into,
    mamba3_backward_adt_product_into,
    mamba3_backward_s17_state_into,
    mamba3_backward_s17_operands_into,
    mamba3_backward_join_s16_s17_into,
    mamba3_backward_s15_only_into,
    mamba3_backward_rotary_only_into,
    mamba3_backward_dacs_to_adt_into,
    mamba3_backward_join_two_into,
    mamba3_backward_a_heavy_tail_into,
    mamba3_backward_bcnorm_into,
    mamba3_backward_pack_in_proj_into,
    mamba3_backward_block_norm_into,
)
from mamba.impl.modeling.modeling_mamba import mamba_download, mamba_upload, mamba_zeros
from mamba.impl.modules.mamba3_prefill_backward import (
    IDN_M3_BWD_SKIP_DEAD,
    Mamba3DeviceGradients,
    Mamba3PrefillGradients,
    _AfnM3Arena,
    _m3_scratch,
    _mtick,
)

# TOMBSTONE: MOJOLEARN_AUTOGRAD_MAMBA3_OFF (the opaque M3_BLOCK rule, tag 72, and the hand Mamba-3 prefill backward walk it called; superseded by the stage rules below) deleted 2026-10-19 by bc0a9a24e; code recoverable at ba37327dc^.
# Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_MAMBA3_OFF.patch


def _check_removed_defines() -> Bool:
    """Refusals for the retired Mamba-3 hand backward (lane ag-mamba3-tape, docs/TOMBSTONES.md). They live here, next to the
    code they named, not in core/six_lane_experiment_guards.mojo: a line there marks every binding that imports core as
    changed."""
    comptime assert not is_defined["MOJOLEARN_AUTOGRAD_MAMBA3_OFF"](), "removed 2026-10-19 (lane/ag-mamba3-tape, bc0a9a24e): the hand Mamba-3 prefill backward (mamba3_prefill_backward_on) and the opaque M3_BLOCK rule are deleted; the stage rules in autograd/rules_mamba3.mojo are the only device backward, same kernels in the same order. Restore with experiments/removed/MOJOLEARN_AUTOGRAD_MAMBA3_OFF.patch"
    return True


comptime M3_REMOVED_DEFINES_OK = _check_removed_defines()

# Tags: LayerNorm 1, recurrent 16-18, transformer 32-47, Mamba-1/2 48-63,
# LM 64-71, Mamba-3 72-87 (72 the retired opaque block rule; 73-78 the stage
# rules; 79-87 free), Samba stack 88-95 (autograd/rules_samba.mojo).
comptime M3_NORM = 73
comptime M3_PROJ_IN = 74
comptime M3_SSD = 75
comptime M3_GATE = 76
comptime M3_PROJ_OUT = 77
#: y = x + O recorded as a launch-free node (see the header)
comptime M3_RESID = 78
comptime M3_TAG_FIRST = M3_NORM
comptime M3_TAG_LAST = 87
#: the nine weights in Mamba3DeviceWeights / registry order
comptime M3_BLOCK_WEIGHTS = 9


def mamba3_supports(tag: Int) -> Bool:
    """Whether `tag` is one of the Mamba-3 stage rules."""
    return tag >= M3_NORM and tag <= M3_RESID


def mamba3_writes_total(tag: Int, edge: Int) -> Bool:
    """The joins (header): x by its norm (D5, the residual term included), P
    by the SSD rule's pack; the gate rule leaves its P terms in the scratch
    and the residual node writes nothing."""
    if tag == M3_NORM or tag == M3_RESID or tag == M3_SSD:
        return edge == 0
    if tag == M3_GATE:
        return edge == 1
    return False


# ------------------------------------------------------------------ helpers
@always_inline
def _fpb(buf: DeviceBuffer[DType.float32]) -> FP:
    """A device buffer's address as the tape's pointer type."""
    return FP(unsafe_from_address=Int(buf.unsafe_ptr()))


@always_inline
def _m3_view(ctx: DeviceContext, p: FP, n: Int) -> DeviceBuffer[DType.float32]:
    """A non-owning view of n floats at p."""
    return DeviceBuffer[DType.float32](ctx, p.unsafe_origin_cast[MutAnyOrigin](), n if n > 0 else 1, owning=False)


def _m3_same(p: FP, buf: DeviceBuffer[DType.float32], what: String) raises:
    # The rules launch on the block's own buffers; the tape must hand them
    # exactly those (mamba3_block_record records them so).
    if Int(p) != Int(buf.unsafe_ptr()):
        raise Error(String("mamba3 backward (autograd): ") + what + " is not the block's buffer")


# ------------------------------------------------------------------ per block state
struct Mamba3TapeScratch(Movable):
    """Every intermediate the hand walk allocates for one block backward
    (other than the four node gradients of Mamba3TapeBlock and the ten
    gradients the tape hands over), allocated as the walk allocates them:
    zeroed, views of the pass's own arena under
    MOJOLEARN_AFN_MAMBA3_BWD_ARENA. Field names are the walk's locals."""
    var arena: _AfnM3Arena
    var workspace: DeviceBuffer[DType.float32]
    var ones: DeviceBuffer[DType.float32]
    # D2
    var d_z: DeviceBuffer[DType.float32]
    var d_v: DeviceBuffer[DType.float32]
    var d_qkdot: DeviceBuffer[DType.float32]
    var d_d_product: DeviceBuffer[DType.float32]
    # D3
    var d_b_qk: DeviceBuffer[DType.float32]
    var d_c_qk: DeviceBuffer[DType.float32]
    var d_b_bias_qk: DeviceBuffer[DType.float32]
    var d_c_bias_qk: DeviceBuffer[DType.float32]
    var d_gamma_qk: DeviceBuffer[DType.float32]
    var d_dt_qk: DeviceBuffer[DType.float32]
    var d_trap_qk: DeviceBuffer[DType.float32]
    var d_q_s16: DeviceBuffer[DType.float32]
    var d_ks_s16: DeviceBuffer[DType.float32]
    var d_v_s16: DeviceBuffer[DType.float32]
    var d_krot_s15: DeviceBuffer[DType.float32]
    var d_scale_s15: DeviceBuffer[DType.float32]
    var d_value_total: DeviceBuffer[DType.float32]
    var d_gamma_scale: DeviceBuffer[DType.float32]
    var d_beta_scale: DeviceBuffer[DType.float32]
    var d_qraw_rot: DeviceBuffer[DType.float32]
    var d_kraw_rot: DeviceBuffer[DType.float32]
    var d_theta_rot: DeviceBuffer[DType.float32]
    var d_b_total: DeviceBuffer[DType.float32]
    var d_c_total: DeviceBuffer[DType.float32]
    var d_gamma_total: DeviceBuffer[DType.float32]
    var d_dt_total: DeviceBuffer[DType.float32]
    var d_trap_total: DeviceBuffer[DType.float32]
    var d_angle_rate: DeviceBuffer[DType.float32]
    var d_angle_raw: DeviceBuffer[DType.float32]
    var d_dt_angle: DeviceBuffer[DType.float32]
    var d_dt_available: DeviceBuffer[DType.float32]
    var d_dt_raw: DeviceBuffer[DType.float32]
    var d_dt_bias_rows: DeviceBuffer[DType.float32]
    var d_dt_bias: DeviceBuffer[DType.float32]
    var d_seg: DeviceBuffer[DType.float32]
    var d_adt_seg: DeviceBuffer[DType.float32]
    var d_a_seg: DeviceBuffer[DType.float32]
    var d_dt_seg: DeviceBuffer[DType.float32]
    var d_dt_with_seg: DeviceBuffer[DType.float32]
    var d_state_direct: DeviceBuffer[DType.float32]
    var d_state_total: DeviceBuffer[DType.float32]
    var d_initial_state: DeviceBuffer[DType.float32]
    var d_q17: DeviceBuffer[DType.float32]
    var d_dacs_read17: DeviceBuffer[DType.float32]
    var d_k17: DeviceBuffer[DType.float32]
    var d_v17: DeviceBuffer[DType.float32]
    var d_dacs_rec17: DeviceBuffer[DType.float32]
    var d_q_join: DeviceBuffer[DType.float32]
    var d_k_join: DeviceBuffer[DType.float32]
    var d_v_join: DeviceBuffer[DType.float32]
    var d_dacs_join: DeviceBuffer[DType.float32]
    var d_krot_join: DeviceBuffer[DType.float32]
    var d_scale_join: DeviceBuffer[DType.float32]
    var d_qraw_join: DeviceBuffer[DType.float32]
    var d_kraw_join: DeviceBuffer[DType.float32]
    var d_theta_join: DeviceBuffer[DType.float32]
    var d_adt_from_dacs: DeviceBuffer[DType.float32]
    var d_adt_join: DeviceBuffer[DType.float32]
    var d_a_join: DeviceBuffer[DType.float32]
    var d_dt_join_adt: DeviceBuffer[DType.float32]
    var d_dt_join_scratch: DeviceBuffer[DType.float32]
    var zero_dt: DeviceBuffer[DType.float32]
    var d_a_raw_join: DeviceBuffer[DType.float32]
    var d_b_join: DeviceBuffer[DType.float32]
    var d_c_join: DeviceBuffer[DType.float32]
    var d_gamma_join: DeviceBuffer[DType.float32]
    var d_dt_join_current: DeviceBuffer[DType.float32]
    var d_trap_join: DeviceBuffer[DType.float32]
    var d_beta_join: DeviceBuffer[DType.float32]
    var d_angle_rate_join: DeviceBuffer[DType.float32]
    var d_angle_raw_join: DeviceBuffer[DType.float32]
    var d_dt_angle_join: DeviceBuffer[DType.float32]
    var d_dt_join_base: DeviceBuffer[DType.float32]
    var d_dt_join_available: DeviceBuffer[DType.float32]
    var d_dt_raw_join: DeviceBuffer[DType.float32]
    var d_dt_bias_rows_join: DeviceBuffer[DType.float32]
    var d_b_raw: DeviceBuffer[DType.float32]
    var d_c_raw: DeviceBuffer[DType.float32]
    var d_bw_rows: DeviceBuffer[DType.float32]
    var d_cw_rows: DeviceBuffer[DType.float32]
    # D5
    var d_norm_w_rows: DeviceBuffer[DType.float32]

    def __init__(out self, ctx: DeviceContext, b: Int, l: Int, dims: Mamba3Dims, nc: Int) raises:
        var arena = _AfnM3Arena()
        var own = arena.own()
        var m = b * l
        var nh = dims.nheads
        var tail_cells = m * dims.d_inner
        var head_cells = m * nh
        var state_cells = m * nh * M3_D_STATE
        var theta_cells = m * nh * M3_NUM_ROPE_ANGLES
        var seg_cells = b * nc * nh * M3_CHUNK_SIZE * M3_CHUNK_SIZE
        var chunk_state_cells = b * nc * nh * M3_HEADDIM * M3_D_STATE
        var initial_state_cells = b * nh * M3_HEADDIM * M3_D_STATE
        self.workspace = mamba_zeros(ctx, mamba3_backward_workspace_max_floats(dims, m))
        var ones = _m3_scratch(own, ctx, mamba3_backward_ones_floats(m))
        ones.enqueue_fill(Float32(1.0))
        self.ones = ones^
        self.d_z = _m3_scratch(own, ctx, tail_cells)
        self.d_v = _m3_scratch(own, ctx, tail_cells)
        self.d_qkdot = _m3_scratch(own, ctx, head_cells)
        self.d_d_product = _m3_scratch(own, ctx, head_cells)
        self.d_b_qk = _m3_scratch(own, ctx, state_cells)
        self.d_c_qk = _m3_scratch(own, ctx, state_cells)
        self.d_b_bias_qk = _m3_scratch(own, ctx, state_cells)
        self.d_c_bias_qk = _m3_scratch(own, ctx, state_cells)
        self.d_gamma_qk = _m3_scratch(own, ctx, head_cells)
        self.d_dt_qk = _m3_scratch(own, ctx, head_cells)
        self.d_trap_qk = _m3_scratch(own, ctx, head_cells)
        self.d_q_s16 = _m3_scratch(own, ctx, state_cells)
        self.d_ks_s16 = _m3_scratch(own, ctx, state_cells)
        self.d_v_s16 = _m3_scratch(own, ctx, tail_cells)
        self.d_krot_s15 = _m3_scratch(own, ctx, state_cells)
        self.d_scale_s15 = _m3_scratch(own, ctx, head_cells)
        self.d_value_total = _m3_scratch(own, ctx, tail_cells)
        self.d_gamma_scale = _m3_scratch(own, ctx, head_cells)
        self.d_beta_scale = _m3_scratch(own, ctx, head_cells)
        self.d_qraw_rot = _m3_scratch(own, ctx, state_cells)
        self.d_kraw_rot = _m3_scratch(own, ctx, state_cells)
        self.d_theta_rot = _m3_scratch(own, ctx, theta_cells)
        self.d_b_total = _m3_scratch(own, ctx, state_cells)
        self.d_c_total = _m3_scratch(own, ctx, state_cells)
        self.d_gamma_total = _m3_scratch(own, ctx, head_cells)
        self.d_dt_total = _m3_scratch(own, ctx, head_cells)
        self.d_trap_total = _m3_scratch(own, ctx, head_cells)
        self.d_angle_rate = _m3_scratch(own, ctx, theta_cells)
        self.d_angle_raw = _m3_scratch(own, ctx, m * M3_NUM_ROPE_ANGLES)
        self.d_dt_angle = _m3_scratch(own, ctx, head_cells)
        self.d_dt_available = _m3_scratch(own, ctx, head_cells)
        self.d_dt_raw = _m3_scratch(own, ctx, head_cells)
        self.d_dt_bias_rows = _m3_scratch(own, ctx, head_cells)
        self.d_dt_bias = _m3_scratch(own, ctx, nh)
        self.d_seg = _m3_scratch(own, ctx, seg_cells)
        self.d_adt_seg = _m3_scratch(own, ctx, head_cells)
        self.d_a_seg = _m3_scratch(own, ctx, head_cells)
        self.d_dt_seg = _m3_scratch(own, ctx, head_cells)
        self.d_dt_with_seg = _m3_scratch(own, ctx, head_cells)
        self.d_state_direct = _m3_scratch(own, ctx, chunk_state_cells)
        self.d_state_total = _m3_scratch(own, ctx, chunk_state_cells)
        self.d_initial_state = _m3_scratch(own, ctx, initial_state_cells)
        self.d_q17 = _m3_scratch(own, ctx, state_cells)
        self.d_dacs_read17 = _m3_scratch(own, ctx, head_cells)
        self.d_k17 = _m3_scratch(own, ctx, state_cells)
        self.d_v17 = _m3_scratch(own, ctx, tail_cells)
        self.d_dacs_rec17 = _m3_scratch(own, ctx, head_cells)
        self.d_q_join = _m3_scratch(own, ctx, state_cells)
        self.d_k_join = _m3_scratch(own, ctx, state_cells)
        self.d_v_join = _m3_scratch(own, ctx, tail_cells)
        self.d_dacs_join = _m3_scratch(own, ctx, head_cells)
        self.d_krot_join = _m3_scratch(own, ctx, state_cells)
        self.d_scale_join = _m3_scratch(own, ctx, head_cells)
        self.d_qraw_join = _m3_scratch(own, ctx, state_cells)
        self.d_kraw_join = _m3_scratch(own, ctx, state_cells)
        self.d_theta_join = _m3_scratch(own, ctx, theta_cells)
        self.d_adt_from_dacs = _m3_scratch(own, ctx, head_cells)
        self.d_adt_join = _m3_scratch(own, ctx, head_cells)
        self.d_a_join = _m3_scratch(own, ctx, head_cells)
        self.d_dt_join_adt = _m3_scratch(own, ctx, head_cells)
        self.d_dt_join_scratch = _m3_scratch(own, ctx, head_cells)
        self.zero_dt = _m3_scratch(own, ctx, head_cells)
        self.d_a_raw_join = _m3_scratch(own, ctx, head_cells)
        self.d_b_join = _m3_scratch(own, ctx, state_cells)
        self.d_c_join = _m3_scratch(own, ctx, state_cells)
        self.d_gamma_join = _m3_scratch(own, ctx, head_cells)
        self.d_dt_join_current = _m3_scratch(own, ctx, head_cells)
        self.d_trap_join = _m3_scratch(own, ctx, head_cells)
        self.d_beta_join = _m3_scratch(own, ctx, head_cells)
        self.d_angle_rate_join = _m3_scratch(own, ctx, theta_cells)
        self.d_angle_raw_join = _m3_scratch(own, ctx, m * M3_NUM_ROPE_ANGLES)
        self.d_dt_angle_join = _m3_scratch(own, ctx, head_cells)
        self.d_dt_join_base = _m3_scratch(own, ctx, head_cells)
        self.d_dt_join_available = _m3_scratch(own, ctx, head_cells)
        self.d_dt_raw_join = _m3_scratch(own, ctx, head_cells)
        self.d_dt_bias_rows_join = _m3_scratch(own, ctx, head_cells)
        self.d_b_raw = _m3_scratch(own, ctx, m * M3_D_STATE)
        self.d_c_raw = _m3_scratch(own, ctx, m * M3_D_STATE)
        self.d_bw_rows = _m3_scratch(own, ctx, m * M3_D_STATE)
        self.d_cw_rows = _m3_scratch(own, ctx, m * M3_D_STATE)
        self.d_norm_w_rows = _m3_scratch(own, ctx, m * dims.d_model)
        self.arena = arena^


struct Mamba3TapeBlock(Movable):
    """One recorded block's backward state: the gradient storage of its four
    interior nodes (NRM, P, SK, G; the walk's d_norm, d_in_proj, d_skip,
    d_gate, zeroed as it zeroes them), the scratch its rules share (present
    from M3_PROJ_OUT to M3_NORM), and the MOJOLEARN_MAMBA_TIMING clock.
    The driver keeps it alive, at a fixed address, through the backward."""
    var d_norm: DeviceBuffer[DType.float32]
    var d_in_proj: DeviceBuffer[DType.float32]
    var d_skip: DeviceBuffer[DType.float32]
    var d_gate: DeviceBuffer[DType.float32]
    var scratch: Optional[Mamba3TapeScratch]
    var tk: Int

    def __init__(out self, ctx: DeviceContext, b: Int, l: Int, dims: Mamba3Dims) raises:
        var m = b * l
        self.d_norm = mamba_zeros[False](ctx, m * dims.d_model)
        self.d_in_proj = mamba_zeros[False](ctx, m * dims.d_in_proj())
        self.d_skip = mamba_zeros[False](ctx, m * dims.d_inner)
        self.d_gate = mamba_zeros[False](ctx, m * dims.d_inner)
        self.scratch = None
        self.tk = 0


comptime M3WeightsRef = MutPointer[Mamba3DeviceWeights, MutUntrackedOrigin]
comptime M3StagesRef = MutPointer[Mamba3DeviceStages, MutUntrackedOrigin]
comptime M3BlockRef = MutPointer[Mamba3TapeBlock, MutUntrackedOrigin]


def _saved(slot: Int, residual_grad: FP) -> Args:
    """A stage node's slots: the Samba slot (i11) and the gradient arriving
    at the block's output (p2; the norm rule adds it into x's gradient)."""
    var a = Args()
    a.i11 = slot
    a.p2 = residual_grad
    return a^


def mamba3_block_record[E: Exec](
    mut tape: Tape,
    mut ex: E,
    x: Tensor,
    params: List[Tensor],
    mut st: Mamba3DeviceStages,
    mut blk: Mamba3TapeBlock,
    out_grad: FP,
    b: Int,
    l: Int,
    slot: Int,
) raises -> Tensor:
    """Record one Mamba-3 block forward that already ran (mamba3_block_forward
    from the zero state into `st`) as its five stage nodes and the launch-free
    residual (header). `params` are the nine weight leaves in registry order;
    `out_grad` holds the gradient at the block output (the caller's d_output,
    or the storage the next consumer writes). Returns the block output
    (st.residual_out), whose gradient storage is `out_grad`."""
    comptime assert M3_REMOVED_DEFINES_OK, "retired Mamba-3 backward define"
    if len(params) != M3_BLOCK_WEIGHTS:
        raise Error("mamba3 backward (autograd): a block takes its nine weights")
    var dims = st.dims.copy()
    var m = b * l
    var dm = dims.d_model
    var di = dims.d_inner
    var dip = dims.d_in_proj()
    var sv = _saved(slot, out_grad)
    var norm_w = params[0]
    var w_in = params[1]
    var dt_bias = params[2]
    var bnorm_w = params[3]
    var cnorm_w = params[4]
    var b_bias = params[5]
    var c_bias = params[6]
    var d_skip = params[7]
    var w_out = params[8]
    # P's consumers SSD then GATE: the gate rule leaves its P terms, the SSD
    # rule packs P's whole gradient (header)
    var nrm = tape.custom_into(ex, [x, norm_w], _fpb(st.norm_out), _fpb(blk.d_norm), m, dm, M3_NORM, sv)
    var p = tape.custom_into(ex, [nrm, w_in], _fpb(st.in_proj), _fpb(blk.d_in_proj), m, dip, M3_PROJ_IN, sv)
    var sk = tape.custom_into(
        ex, [p, dt_bias, bnorm_w, cnorm_w, b_bias, c_bias], _fpb(st.skip_out), _fpb(blk.d_skip),
        m, di, M3_SSD, sv,
    )
    var g = tape.custom_into(ex, [sk, p, d_skip], _fpb(st.gate_out), _fpb(blk.d_gate), m, di, M3_GATE, sv)
    var o = tape.custom_into(ex, [g, w_out], _fpb(st.out_proj), out_grad, m, dm, M3_PROJ_OUT, sv)
    return tape.custom_into_shared(ex, [x, o], _fpb(st.residual_out), o, m, dm, M3_RESID, sv)


# ------------------------------------------------------------------ the rules
struct Mamba3Rules(BackwardRules):
    """The stage rules of ONE recorded Mamba-3 block (M3_NORM .. M3_RESID).

    Holds pointers to the block's weights, its recorded forward stages and
    its Mamba3TapeBlock; all must outlive the backward."""

    var ctx: DeviceContext
    var w: M3WeightsRef
    var st: M3StagesRef
    var blk: M3BlockRef
    var b: Int
    var l: Int
    var ton: Bool

    def __init__(
        out self, ctx: DeviceContext, w: M3WeightsRef, st: M3StagesRef, blk: M3BlockRef,
        b: Int, l: Int, ton: Bool,
    ):
        self.ctx = ctx.copy()
        self.w = w
        self.st = st
        self.blk = blk
        self.b = b
        self.l = l
        self.ton = ton

    def zeroed_destination(self, tag: Int, saved: Args, edge: Int) -> Bool:
        # Never: every destination is stored whole (header).
        return False

    def writes_total(self, tag: Int, saved: Args, edge: Int) -> Bool:
        return mamba3_writes_total(tag, edge)

    def supports(self, tag: Int) -> Bool:
        return mamba3_supports(tag)

    def apply[E: Exec](
        self, mut ex: E, tag: Int, saved: Args, output: FP, upstream: FP,
        inputs: List[FP], gradients: List[FP], rows: List[Int], cols: List[Int],
    ) raises:
        # Every launch goes on self.ctx, the context `ex` wraps.
        if tag == M3_PROJ_OUT:
            self._proj_out(upstream, gradients, rows, cols)
        elif tag == M3_GATE:
            self._gate(upstream, gradients, rows, cols)
        elif tag == M3_SSD:
            self._ssd(upstream, gradients, rows, cols)
        elif tag == M3_PROJ_IN:
            self._proj_in(upstream, gradients, rows, cols)
        elif tag == M3_NORM:
            self._norm(saved, upstream, inputs, gradients, rows, cols)
        elif tag == M3_RESID:
            # no launch: O's gradient is this node's storage, x's is written
            # whole by the block's norm
            if Int(gradients[1]) != Int(upstream):
                raise Error("mamba3 backward (autograd): the residual's branch must share the output's gradient")
        else:
            raise Error("unknown Mamba-3 backward rule")

    def _proj_out(self, upstream: FP, gradients: List[FP], rows: List[Int], cols: List[Int]) raises:
        # D1, the block's first rule: the walk's scratch is allocated here.
        ref w = self.w[]
        ref st = self.st[]
        ref blk = self.blk[]
        var ctx = self.ctx.copy()
        var dims = w.dims.copy()
        var m = self.b * self.l
        _m3_same(gradients[0], blk.d_gate, "the gate output gradient")
        if blk.scratch:
            raise Error("mamba3 backward (autograd): the block's scratch is still held")
        blk.tk = Int(perf_counter_ns())
        blk.scratch = Mamba3TapeScratch(ctx, self.b, self.l, dims, st.nc)
        ref S = blk.scratch.value()
        var d_output = _m3_view(ctx, upstream, m * dims.d_model)
        var d_weight = _m3_view(ctx, gradients[1], rows[1] * cols[1])
        mamba3_backward_proj_a_into(ctx, blk.d_gate, d_output, w.w_out, S.workspace, PROJ3_OUT, dims, m)
        _mtick(ctx, self.ton, blk.tk, "proj_a")
        mamba3_backward_proj_b_into(ctx, d_weight, d_output, st.gate_out, S.workspace, PROJ3_OUT, dims, m)
        _mtick(ctx, self.ton, blk.tk, "proj_b")

    def _gate(self, upstream: FP, gradients: List[FP], rows: List[Int], cols: List[Int]) raises:
        # D2. d_skip is SK's gradient storage; d_z and d_v (P's terms) and
        # d_qkdot stay in the scratch for the SSD rule; P's edge is left.
        ref w = self.w[]
        ref st = self.st[]
        ref blk = self.blk[]
        var ctx = self.ctx.copy()
        var dims = w.dims.copy()
        var m = self.b * self.l
        _m3_same(upstream, blk.d_gate, "the gate output gradient")
        _m3_same(gradients[0], blk.d_skip, "the skip output gradient")
        ref S = blk.scratch.value()
        var d_d = _m3_view(ctx, gradients[2], rows[2] * cols[2])
        mamba3_backward_gate_skip_into(
            ctx, blk.d_skip, S.d_z, S.d_v, S.d_qkdot, S.d_d_product, blk.d_gate,
            st.skip_out, st.qkdot, st.in_proj,
            w.d_skip, dims, m,
        )
        _mtick(ctx, self.ton, blk.tk, "gate_skip")
        mamba3_backward_reduce_into(ctx, d_d, S.d_d_product, S.ones, S.workspace, RED3_D, dims, m)
        _mtick(ctx, self.ton, blk.tk, "reduce")

    def _ssd(self, upstream: FP, gradients: List[FP], rows: List[Int], cols: List[Int]) raises:
        # D3 in the walk's order, ending in the pack of P's whole gradient
        # (writes_total). gradients: [P, dt_bias, bnorm_w, cnorm_w, b_bias,
        # c_bias].
        ref w = self.w[]
        ref st = self.st[]
        ref blk = self.blk[]
        var ctx = self.ctx.copy()
        var dims = w.dims.copy()
        var b = self.b
        var l = self.l
        var m = b * l
        var nh = dims.nheads
        var tail_cells = m * dims.d_inner
        var head_cells = m * nh
        var state_cells = m * nh * M3_D_STATE
        _m3_same(upstream, blk.d_skip, "the skip output gradient")
        _m3_same(gradients[0], blk.d_in_proj, "the in_proj output gradient")
        ref S = blk.scratch.value()
        var own = S.arena.own()
        var d_dt_bias_join = _m3_view(ctx, gradients[1], rows[1] * cols[1])
        var d_bw = _m3_view(ctx, gradients[2], rows[2] * cols[2])
        var d_cw = _m3_view(ctx, gradients[3], rows[3] * cols[3])
        var d_bb = _m3_view(ctx, gradients[4], rows[4] * cols[4])
        var d_cb = _m3_view(ctx, gradients[5], rows[5] * cols[5])
        mamba3_backward_qkdot_into(
            ctx, S.d_b_qk, S.d_c_qk, S.d_b_bias_qk, S.d_c_bias_qk, S.d_gamma_qk,
            S.d_dt_qk, S.d_trap_qk,
            S.d_qkdot, st.bcnorm_b, st.bcnorm_c, w.b_bias,
            w.c_bias, st.gamma_work, st.dt_work,
            st.sig_work, dims, m,
        )
        _mtick(ctx, self.ton, blk.tk, "qkdot")
        mamba3_backward_s16_s15_into(
            ctx, S.d_q_s16, S.d_ks_s16, S.d_v_s16, S.d_krot_s15, S.d_scale_s15,
            blk.d_skip, st.rotq_work, st.kscale_work, st.v_work,
            st.seg_l, st.rotk_work, st.scale_work,
            b, l, dims, M3_CHUNK_SIZE,
        )
        _mtick(ctx, self.ton, blk.tk, "s16_s15")
        mamba3_backward_join_rotary_into(
            ctx, S.d_value_total, S.d_gamma_scale, S.d_beta_scale, S.d_qraw_rot,
            S.d_kraw_rot, S.d_theta_rot, S.d_v, S.d_v_s16, S.d_scale_s15,
            S.d_q_s16, S.d_krot_s15, st.bcnorm_b, st.bcnorm_c,
            w.b_bias, w.c_bias, st.theta_out,
            m, dims,
        )
        _mtick(ctx, self.ton, blk.tk, "join_rotary")
        # IDN_M3_BWD_SKIP_DEAD: the walk's pre-join pass (no returned
        # gradient reads it), run only under its _OFF define, as the walk did.
        comptime if not IDN_M3_BWD_SKIP_DEAD:
            mamba3_backward_join_current_into(
                ctx, S.d_b_total, S.d_c_total, S.d_gamma_total, S.d_dt_total, S.d_trap_total,
                S.d_b_qk, S.d_c_qk, S.d_kraw_rot, S.d_qraw_rot, S.d_gamma_qk, S.d_gamma_scale,
                S.d_dt_qk, S.d_trap_qk, S.d_beta_scale, st.dt_work, st.sig_work, b, l, dims,
            )
            _mtick(ctx, self.ton, blk.tk, "join_current")
            comptime if AFN_M3_BWD_CHUNK:
                var afn_sums = _m3_scratch(own, ctx, b * nh * M3_NUM_ROPE_ANGLES * afn_m3_theta_chunks(l))
                mamba3_afn_backward_angle_into(
                    ctx, S.d_angle_rate, S.d_angle_raw, S.d_dt_angle, S.d_theta_rot,
                    st.dt_work, st.in_proj, afn_sums, b, l, dims,
                )
                S.arena.keep.append(afn_sums^)
            else:
                var sums_rot = _m3_scratch(own, ctx, mamba3_angle_suffix_sums_cells(b, l, nh))
                mamba3_backward_angle_into(
                    ctx, S.d_angle_rate, S.d_angle_raw, S.d_dt_angle, S.d_theta_rot,
                    st.dt_work, st.in_proj, b, l, dims, sums_rot,
                )
                S.arena.keep.append(sums_rot^)
            _mtick(ctx, self.ton, blk.tk, "angle")
            mamba3_backward_dt_partial_into(
                ctx, S.d_dt_available, S.d_dt_raw, S.d_dt_bias_rows, S.d_dt_total, S.d_dt_angle,
                st.in_proj, w.dt_bias, m, dims,
            )
            _mtick(ctx, self.ton, blk.tk, "dt_partial")
            mamba3_backward_reduce_into(ctx, S.d_dt_bias, S.d_dt_bias_rows, S.ones, S.workspace, RED3_DT_BIAS, dims, m)
            _mtick(ctx, self.ton, blk.tk, "reduce")
        mamba3_backward_seg_adt_into(
            ctx, S.d_seg, S.d_adt_seg, blk.d_skip, st.rotq_work, st.kscale_work, st.v_work, st.seg_l,
            b, l, dims, M3_CHUNK_SIZE,
        )
        _mtick(ctx, self.ton, blk.tk, "seg_adt")
        comptime if not IDN_M3_BWD_SKIP_DEAD:
            mamba3_backward_adt_product_into(
                ctx, S.d_a_seg, S.d_dt_seg, S.d_dt_with_seg, S.d_adt_seg, st.a_out, st.dt_out,
                S.d_dt_available, head_cells,
            )
            _mtick(ctx, self.ton, blk.tk, "adt_product")
        mamba3_backward_s17_state_into(
            ctx, S.d_state_direct, S.d_state_total, S.d_initial_state, blk.d_skip, st.rotq_work, st.dacs,
            b, l, dims, M3_CHUNK_SIZE,
        )
        _mtick(ctx, self.ton, blk.tk, "s17_state")
        mamba3_backward_s17_operands_into(
            ctx, S.d_q17, S.d_dacs_read17, S.d_k17, S.d_v17, S.d_dacs_rec17,
            blk.d_skip, st.rotq_work, st.kscale_work, st.v_work, st.dacs, st.pass_states, S.d_state_total,
            b, l, dims, M3_CHUNK_SIZE,
        )
        _mtick(ctx, self.ton, blk.tk, "s17_operands")
        mamba3_backward_join_s16_s17_into(
            ctx, S.d_q_join, S.d_k_join, S.d_v_join, S.d_dacs_join,
            S.d_q_s16, S.d_q17, S.d_ks_s16, S.d_k17, S.d_value_total, S.d_v17, S.d_dacs_read17, S.d_dacs_rec17,
            state_cells, tail_cells, head_cells,
        )
        _mtick(ctx, self.ton, blk.tk, "join_s16_s17")
        mamba3_backward_s15_only_into(
            ctx, S.d_krot_join, S.d_scale_join, S.d_k_join, st.rotk_work, st.scale_work, head_cells,
        )
        _mtick(ctx, self.ton, blk.tk, "s15_only")
        mamba3_backward_rotary_only_into(
            ctx, S.d_qraw_join, S.d_kraw_join, S.d_theta_join, S.d_q_join, S.d_krot_join,
            st.bcnorm_b, st.bcnorm_c, w.b_bias, w.c_bias, st.theta_out,
            m * nh * (M3_D_STATE // 2), nh,
        )
        _mtick(ctx, self.ton, blk.tk, "rotary_only")
        mamba3_backward_dacs_to_adt_into(ctx, S.d_adt_from_dacs, S.d_dacs_join, b, l, nh, M3_CHUNK_SIZE)
        _mtick(ctx, self.ton, blk.tk, "dacs_to_adt")
        mamba3_backward_join_two_into(ctx, S.d_adt_join, S.d_adt_seg, S.d_adt_from_dacs, head_cells)
        _mtick(ctx, self.ton, blk.tk, "join_two")
        mamba3_backward_adt_product_into(
            ctx, S.d_a_join, S.d_dt_join_adt, S.d_dt_join_scratch, S.d_adt_join, st.a_out, st.dt_out,
            S.zero_dt, head_cells,
        )
        _mtick(ctx, self.ton, blk.tk, "adt_product")
        mamba3_backward_a_heavy_tail_into(ctx, S.d_a_raw_join, S.d_a_join, st.in_proj, m, dims)
        _mtick(ctx, self.ton, blk.tk, "a_heavy_tail")
        ctx.enqueue_copy(dst_buf=S.d_beta_join, src_buf=S.d_scale_join)
        mamba3_backward_join_current_into(
            ctx, S.d_b_join, S.d_c_join, S.d_gamma_join, S.d_dt_join_current, S.d_trap_join,
            S.d_b_qk, S.d_c_qk, S.d_kraw_join, S.d_qraw_join, S.d_gamma_qk, S.d_scale_join,
            S.d_dt_qk, S.d_trap_qk, S.d_beta_join, st.dt_work, st.sig_work, b, l, dims,
        )
        _mtick(ctx, self.ton, blk.tk, "join_current")
        comptime if AFN_M3_BWD_CHUNK:
            var afn_sums = _m3_scratch(own, ctx, b * nh * M3_NUM_ROPE_ANGLES * afn_m3_theta_chunks(l))
            mamba3_afn_backward_angle_into(
                ctx, S.d_angle_rate_join, S.d_angle_raw_join, S.d_dt_angle_join, S.d_theta_join,
                st.dt_work, st.in_proj, afn_sums, b, l, dims,
            )
            S.arena.keep.append(afn_sums^)
        else:
            var sums_join = _m3_scratch(own, ctx, mamba3_angle_suffix_sums_cells(b, l, nh))
            mamba3_backward_angle_into(
                ctx, S.d_angle_rate_join, S.d_angle_raw_join, S.d_dt_angle_join, S.d_theta_join,
                st.dt_work, st.in_proj, b, l, dims, sums_join,
            )
            S.arena.keep.append(sums_join^)
        _mtick(ctx, self.ton, blk.tk, "angle")
        mamba3_backward_join_two_into(ctx, S.d_dt_join_base, S.d_dt_join_current, S.d_dt_join_adt, head_cells)
        _mtick(ctx, self.ton, blk.tk, "join_two")
        mamba3_backward_dt_partial_into(
            ctx, S.d_dt_join_available, S.d_dt_raw_join, S.d_dt_bias_rows_join, S.d_dt_join_base,
            S.d_dt_angle_join, st.in_proj, w.dt_bias, m, dims,
        )
        _mtick(ctx, self.ton, blk.tk, "dt_partial")
        mamba3_backward_reduce_into(
            ctx, d_dt_bias_join, S.d_dt_bias_rows_join, S.ones, S.workspace, RED3_DT_BIAS, dims, m
        )
        _mtick(ctx, self.ton, blk.tk, "reduce")
        mamba3_backward_bcnorm_into(ctx, S.d_b_raw, S.d_bw_rows, S.d_b_join, st.in_proj, w.bnorm_w, m, dims, dims.col_b())
        _mtick(ctx, self.ton, blk.tk, "bcnorm")
        mamba3_backward_bcnorm_into(ctx, S.d_c_raw, S.d_cw_rows, S.d_c_join, st.in_proj, w.cnorm_w, m, dims, dims.col_c())
        _mtick(ctx, self.ton, blk.tk, "bcnorm")
        mamba3_backward_reduce_into(ctx, d_bw, S.d_bw_rows, S.ones, S.workspace, RED3_BNORM_W, dims, m)
        mamba3_backward_reduce_into(ctx, d_cw, S.d_cw_rows, S.ones, S.workspace, RED3_CNORM_W, dims, m)
        _mtick(ctx, self.ton, blk.tk, "reduce")
        mamba3_backward_reduce_into(ctx, d_bb, S.d_b_join, S.ones, S.workspace, RED3_B_BIAS, dims, m)
        mamba3_backward_reduce_into(ctx, d_cb, S.d_c_join, S.ones, S.workspace, RED3_C_BIAS, dims, m)
        _mtick(ctx, self.ton, blk.tk, "reduce")
        mamba3_backward_pack_in_proj_into(
            ctx, blk.d_in_proj, S.d_z, S.d_v_join, S.d_b_raw, S.d_c_raw, S.d_dt_raw_join, S.d_a_raw_join,
            S.d_trap_join, S.d_angle_raw_join, m, dims,
        )
        _mtick(ctx, self.ton, blk.tk, "pack_in_proj")

    def _proj_in(self, upstream: FP, gradients: List[FP], rows: List[Int], cols: List[Int]) raises:
        # D4
        ref w = self.w[]
        ref st = self.st[]
        ref blk = self.blk[]
        var ctx = self.ctx.copy()
        var dims = w.dims.copy()
        var m = self.b * self.l
        _m3_same(upstream, blk.d_in_proj, "the in_proj output gradient")
        _m3_same(gradients[0], blk.d_norm, "the norm output gradient")
        ref S = blk.scratch.value()
        var d_w_in = _m3_view(ctx, gradients[1], rows[1] * cols[1])
        mamba3_backward_proj_a_into(ctx, blk.d_norm, blk.d_in_proj, w.w_in, S.workspace, PROJ3_IN, dims, m)
        _mtick(ctx, self.ton, blk.tk, "proj_a")
        mamba3_backward_proj_b_into(ctx, d_w_in, blk.d_in_proj, st.norm_out, S.workspace, PROJ3_IN, dims, m)
        _mtick(ctx, self.ton, blk.tk, "proj_b")

    def _norm(
        self, saved: Args, upstream: FP, inputs: List[FP], gradients: List[FP], rows: List[Int], cols: List[Int],
    ) raises:
        # D5 with the gradient at the block's output (saved.p2) as its
        # d_output, so x's whole gradient lands in gradients[0]
        # (writes_total); then the walk's closing wait, and the scratch goes.
        ref w = self.w[]
        ref st = self.st[]
        ref blk = self.blk[]
        var ctx = self.ctx.copy()
        var dims = w.dims.copy()
        var m = self.b * self.l
        var dm = dims.d_model
        if rows[0] * cols[0] != m * dm:
            raise Error("mamba3 backward (autograd): the block input must hold [B, L, d_model]")
        _m3_same(upstream, blk.d_norm, "the norm output gradient")
        ref S = blk.scratch.value()
        var d_x = _m3_view(ctx, gradients[0], m * dm)
        var d_output = _m3_view(ctx, saved.p2, m * dm)
        var x = _m3_view(ctx, inputs[0], m * dm)
        var d_norm_w = _m3_view(ctx, gradients[1], rows[1] * cols[1])
        mamba3_backward_block_norm_into(
            ctx, d_x, S.d_norm_w_rows, blk.d_norm, d_output, x, st.norm_sumsq, w.norm_w, m, dims
        )
        _mtick(ctx, self.ton, blk.tk, "block_norm")
        mamba3_backward_reduce_into(ctx, d_norm_w, S.d_norm_w_rows, S.ones, S.workspace, RED3_NORM_W, dims, m)
        _mtick(ctx, self.ton, blk.tk, "reduce")
        # the walk's closing synchronize: every launch above read the scratch
        ctx.synchronize()
        S.arena.close()
        blk.scratch = None


# ------------------------------------------------------------------ drivers
def mamba3_tape_backward_on(
    ctx: DeviceContext,
    mut w: Mamba3DeviceWeights,
    mut st: Mamba3DeviceStages,
    mut x: DeviceBuffer[DType.float32],
    mut d_output: DeviceBuffer[DType.float32],
    b: Int,
    l: Int,
    dims: Mamba3Dims,
    ton: Bool,
) raises -> Mamba3DeviceGradients:
    """The zero-state prefill VJP of one Mamba-3 block on device-resident
    operands, assembled by the tape: `st` and `x` are the forward's (recorded
    by `mamba3_block_forward` on `w` from the zero state), `d_output` the
    gradient at the block output. Synchronized on return; the ten gradients
    stay on the device."""
    var m = b * l
    var dm = dims.d_model
    var di = dims.d_inner
    var dip = dims.d_in_proj()
    var nh = dims.nheads
    # the ten outputs, allocated as the walk allocated them
    var g = Mamba3DeviceGradients(
        mamba_zeros[False](ctx, m * dm),
        mamba_zeros[False](ctx, dm),
        mamba_zeros[False](ctx, dip * dm),
        mamba_zeros[False](ctx, nh),
        mamba_zeros[False](ctx, M3_D_STATE),
        mamba_zeros[False](ctx, M3_D_STATE),
        mamba_zeros[False](ctx, nh * M3_D_STATE),
        mamba_zeros[False](ctx, nh * M3_D_STATE),
        mamba_zeros[False](ctx, nh),
        mamba_zeros[False](ctx, dm * di),
    )
    var blk = Mamba3TapeBlock(ctx, b, l, dims)
    var ex = DeviceExec(ctx)
    var tape = Tape()
    var x_t = tape.variable_into(ex, _fpb(x), _fpb(g.x), m, dm)
    var params = List[Tensor]()
    params.append(tape.variable_into(ex, _fpb(w.norm_w), _fpb(g.block_norm_weight), 1, dm))
    params.append(tape.variable_into(ex, _fpb(w.w_in), _fpb(g.in_proj_weight), dip, dm))
    params.append(tape.variable_into(ex, _fpb(w.dt_bias), _fpb(g.dt_bias), 1, nh))
    params.append(tape.variable_into(ex, _fpb(w.bnorm_w), _fpb(g.B_norm_weight), 1, M3_D_STATE))
    params.append(tape.variable_into(ex, _fpb(w.cnorm_w), _fpb(g.C_norm_weight), 1, M3_D_STATE))
    params.append(tape.variable_into(ex, _fpb(w.b_bias), _fpb(g.B_bias), 1, nh * M3_D_STATE))
    params.append(tape.variable_into(ex, _fpb(w.c_bias), _fpb(g.C_bias), 1, nh * M3_D_STATE))
    params.append(tape.variable_into(ex, _fpb(w.d_skip), _fpb(g.D), 1, nh))
    params.append(tape.variable_into(ex, _fpb(w.w_out), _fpb(g.out_proj_weight), dm, di))
    var y = mamba3_block_record(tape, ex, x_t, params, st, blk, _fpb(d_output), b, l, 0)
    # the seed is the output's own gradient storage, used in place
    var seed = tape.variable(ex, _fpb(d_output), m, dm, False)
    var rules = Mamba3Rules(
        ctx,
        M3WeightsRef(unsafe_from_address=Int(MutPointer(to=w))),
        M3StagesRef(unsafe_from_address=Int(MutPointer(to=st))),
        M3BlockRef(unsafe_from_address=Int(MutPointer(to=blk))),
        b, l, ton,
    )
    try:
        tape.backward_with_rules(ex, y, seed, rules)
    except e:
        # the block's scratch may still be read by queued launches
        ex.sync()
        raise e
    ex.sync()
    tape.release(ex)
    ctx.synchronize()
    _ = rules^
    _ = blk^
    _ = params^
    return g^


def mamba3_tape_backward(
    weights: Mamba3Weights,
    input: List[Float32],
    grad_output: List[Float32],
    b: Int,
    l: Int,
    ctx: DeviceContext,
) raises -> Mamba3PrefillGradients:
    """The zero-state prefill VJP of one Mamba-3 block from host lists: the
    block forward from the zero state, then `mamba3_tape_backward_on`, then
    the ten gradients downloaded."""
    if b <= 0 or l <= 0:
        raise Error("mamba3 backward: B and L must be positive")
    if len(input) != b * l * weights.dims.d_model or len(grad_output) != len(input):
        raise Error("mamba3 backward: input and grad_output lengths must equal B*L*d_model")
    var dims = weights.dims.copy()
    var m = b * l
    var ton = String(getenv("MOJOLEARN_MAMBA_TIMING")) != ""
    var device_weights = Mamba3DeviceWeights(ctx, weights)
    var state = allocate_inference_cache(ctx, b, dims)
    var stages = Mamba3DeviceStages(ctx, b, l, 0, dims)
    var x = mamba_upload(ctx, input)
    var trace = IdentityTrace.disabled()
    mamba3_block_forward(
        ctx, stages, state, device_weights, x, b, l,
        trace, String("m3.backward.tail"),
    )
    # the grad_output finite refusal runs on the uploaded buffer
    var d_output = mamba_upload(ctx, grad_output)
    mamba_refuse_grad_output(ctx, String("mamba3"), d_output, grad_output)
    var gradients = mamba3_tape_backward_on(ctx, device_weights, stages, x, d_output, b, l, dims, ton)
    var result = Mamba3PrefillGradients()
    result.x = mamba_download(ctx, gradients.x, m * dims.d_model)
    result.block_norm_weight = mamba_download(ctx, gradients.block_norm_weight, dims.d_model)
    result.in_proj_weight = mamba_download(ctx, gradients.in_proj_weight, dims.d_in_proj() * dims.d_model)
    result.dt_bias = mamba_download(ctx, gradients.dt_bias, dims.nheads)
    result.B_norm_weight = mamba_download(ctx, gradients.B_norm_weight, M3_D_STATE)
    result.C_norm_weight = mamba_download(ctx, gradients.C_norm_weight, M3_D_STATE)
    result.B_bias = mamba_download(ctx, gradients.B_bias, dims.nheads * M3_D_STATE)
    result.C_bias = mamba_download(ctx, gradients.C_bias, dims.nheads * M3_D_STATE)
    result.D = mamba_download(ctx, gradients.D, dims.nheads)
    result.out_proj_weight = mamba_download(ctx, gradients.out_proj_weight, dims.d_model * dims.d_inner)
    # Explicit ownership extends all GPU buffers through synchronization.
    _ = gradients^
    _ = d_output^
    _ = x^
    _ = stages^
    _ = state^
    _ = device_weights^
    return result^
