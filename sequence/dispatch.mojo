# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`apply[OP]`: the one table from an operation code to its element body, read
by both executors (`sequence/exec.mojo`, `sequence/exec_device.mojo`)."""
from sequence.ops import (
    Args,
    OP_GEMM,
    OP_GEMM_SPLITK,
    OP_BIAS,
    OP_COLSUM,
    OP_CELL_FWD,
    OP_CELL_BWD,
    OP_CELL_FWD_H,
    OP_CELL_BWD_H,
    OP_GEMM_EPI,
    OP_COLSUM_DIV,
    OP_GEMM_EPI_TAIL,
    OP_GATHER_SEQ,
    OP_GATHER_ROWS,
    OP_MSE,
    OP_CE,
    OP_SUM,
    OP_OPT,
    OP_FILL,
    OP_COPY,
    OP_SEQ_OUT,
    OP_SOFTMAX,
    OP_STL,
    OP_VAR_DESIGN,
    OP_COLSCALE,
    OP_CHOLSOLVE,
    OP_ROWSCALE,
    OP_VAR_FORECAST,
    OP_SUB,
    OP_SCALE,
    OP_ACT,
    OP_ACT_BWD,
    OP_MLP_ROWLOSS,
    OP_SUMSQ,
    OP_MLP_BLOSS,
    OP_MLP_L2PART,
    OP_MLP_L2FOLD,
    OP_MLP_ROWPART,
    OP_MLP_PERM,
    OP_MLP_EPOCH_LOSS,
    OP_MLP_LB_PART,
    OP_MLP_LB_FOLD,
    OP_MLP_LB_EPOCH_PART,
    OP_MLP_LB_EPOCH_FOLD,
    OP_MLP_LB_FIRST,
    OP_MLP_LB_LAST,
    OP_SEQ_IOTA,
    OP_PROPHET_PREP,
    OP_L2GRAD,
    OP_DIVS,
    OP_AF_ALPHA,
    OP_AF_ROW,
    OP_AF_COL,
    OP_AF_RMEAN,
    OP_AF_UPDATE_MAT,
    OP_AF_VEC,
    OP_AF_DENOM,
    OP_AF_APPLY,
    OP_SEG_SUMSQ,
    OP_CHUNK_SUMSQ,
    OP_AF_BLK_SUMSQ,
    OP_LAMB_UPD,
    OP_LAMB_RATIO,
    OP_LAMB_APPLY,
    OP_LAMB_BLK,
    OP_LAMB_SEGFOLD,
    OP_LAMB_CLIP,
    OP_LAMB_TRUST,
    OP_LAMB_APPLY_ALL,
    OP_LN_FWD,
    OP_LN_BWD_X,
    OP_LN_BWD_W,
    OP_THETA,
    OP_CROSTON,
    OP_ETS,
    OP_GARCH,
    OP_PROPHET_FEATURES,
    OP_PROPHET_FIT,
    OP_PROPHET_PREDICT,
    OP_PROPHET_FG_PART,
    OP_PROPHET_FG_SUM,
    OP_MOE_ROUTE,
    OP_MOE_HIDDEN,
    OP_MOE_OUT,
    OP_ETS_LIK,
    OP_ETS_INIT,
    OP_VAR_RESID,
    OP_VAR_SIGMA,
    OP_STL_SEAS,
    OP_STL_MA,
    OP_STL_LOESS,
    OP_STL_DESEAS,
    OP_STL_FINISH,
    OP_ONE_HOT,
    OP_PROBA2,
    op_one_hot,
    op_proba2,
    op_gemm,
    op_gemm_splitk,
    op_bias,
    op_colsum,
    op_cell_fwd,
    op_cell_bwd,
    op_cell_fwd_h,
    op_cell_bwd_h,
    op_gather_seq,
    op_gather_rows,
    op_mse,
    op_ce,
    op_sum,
    op_opt,
    op_fill,
    op_seq_iota,
    op_copy,
    op_seq_out,
    op_softmax,
)
from sequence.adafactor import op_af_alpha, op_af_row, op_af_col, op_af_rmean, op_af_update_mat, op_af_vec, op_af_denom, op_af_apply, op_seg_sumsq, op_chunk_sumsq, op_af_blk_sumsq, op_lamb_upd, op_lamb_ratio, op_lamb_apply, op_lamb_blk, op_lamb_segfold, op_lamb_clip, op_lamb_trust, op_lamb_apply_all
from sequence.layernorm import op_ln_bwd_w, op_ln_bwd_x, op_ln_fwd
from sequence.mlp import op_act, op_act_bwd, op_divs, op_gemm_epi, op_gemm_epi_tail, op_l2grad, op_mlp_bloss, op_mlp_rowloss, op_sumsq
from sequence.mlp import op_mlp_epoch_loss, op_mlp_l2fold, op_mlp_l2part, op_mlp_perm, op_mlp_rowpart
from autograd.kernels.linear import op_agk_gemm_epi, op_colsum_div
from sequence.mlp import op_mlp_lb_epoch_fold, op_mlp_lb_epoch_part, op_mlp_lb_fold, op_mlp_lb_part
from sequence.stl import op_stl
from sequence.theta import op_theta
from sequence.croston import op_croston
from sequence.ets import op_ets, op_ets_init, op_ets_lik
from sequence.garch import op_garch
from sequence.moe import op_moe_hidden, op_moe_out, op_moe_route
from sequence.prophet import op_prophet_prep, op_prophet_features, op_prophet_fit, op_prophet_predict, op_prophet_fg_part, op_prophet_fg_sum
from sequence.recurrent_scan import OP_CELL_BWD_SCAN, OP_CELL_FWD_SCAN, op_cell_bwd_scan, op_cell_fwd_scan
from sequence.vecar import op_cholsolve, op_colscale, op_rowscale, op_scale, op_sub, op_var_design, op_var_forecast
from sequence.vecar import op_var_resid, op_var_sigma
from sequence.stl_grid import op_stl_deseas, op_stl_finish, op_stl_loess, op_stl_ma, op_stl_seas
from autograd.ops import OP_AG_FIRST, OP_AG_LAST, apply_autograd
from autograd.ops import OP_AGK_FIRST, OP_AGK_LAST, OP_AGK_GEMM_EPI, apply_autograd_kernels
from autograd.ops import OP_AGF_FIRST, OP_AGF_LAST, apply_autograd_fold
from autograd.data_ops import OP_AGD_FIRST, OP_AGD_LAST, apply_autograd_data
from autograd.optim import OP_AGO_FIRST, OP_AGO_LAST, apply_optim
from autograd.optim import OP_AGH_FIRST, OP_AGH_LAST, apply_optim_fused
from autograd.loss_ops import OP_LS_FIRST, OP_LS_LAST, apply_losses
from autograd.model_ops import OP_AGM_FIRST, OP_AGM_LAST, apply_model_ops
from autograd.pass1_ops import OP_P1_FIRST, OP_P1_LAST, apply_pass1
from autograd.composed_ops import OP_AGC_FIRST, OP_AGC_LAST, apply_composed
from autograd.ops import OP_AG_SOFTMAX_FWD, OP_AG_SOFTMAX_VJP
from autograd.row_ops import OP_ROW_FIRST, OP_ROW_LAST, ROW_COOP, apply_rows, row_softmax_fwd, row_softmax_vjp
from autograd.fold_ops import OP_AGS_FIRST, OP_AGS_LAST, apply_fold_ops
from sequence.gemm_split import OP_GK_FIRST, OP_GK_LAST, OP_GK_SPLIT, op_gk_split
from autograd.kernels.moe import OP_MOEB_FIRST, OP_MOEB_LAST, apply_moe_backward
from autograd.embedding_ops import OP_EMB_FIRST, OP_EMB_LAST, apply_embedding
# lane ag-ho-fused (H2): the recorded loss VJPs' one-hot constant
from autograd.higher_loss_ops import OP_HL_FIRST, OP_HL_LAST, apply_higher_loss
from autograd.cast_ops import OP_CAST_FIRST, OP_CAST_LAST, apply_cast
# lane ag-rankn-core: the rank-N view copy (autograd/rankn_ops.mojo OP_RNC_*, 4100..4109)
from autograd.rankn_ops import OP_RNC_FIRST, OP_RNC_LAST, apply_rankn_core
# >>> lane ag-rankn-fold: rank-N broadcasts, axis reductions, max (4200..4219)
from autograd.rankn_fold_ops import OP_RNF_FIRST, OP_RNF_LAST, apply_rankn_fold
# <<< lane ag-rankn-fold
# lane ag-rankn-bmm: the batched matrix product (docs/plans/AUTOGRAD_RANKN.md 2.3)
from autograd.rankn_bmm_ops import OP_RNB_FIRST, OP_RNB_LAST, apply_rn_bmm
# rank-N layout (lane ag-rankn-layout): the slab copy family, 4400..4409
from autograd.rankn_layout_ops import OP_RNL_FIRST, OP_RNL_LAST, apply_rankn_layout
# ---- lane ag-rankn-conv (AUTOGRAD_RANKN.md 2.8): rank-4 conv2d and pools, 4500..4519
from autograd.rankn_conv_ops import OP_RNV_FIRST, OP_RNV_LAST, apply_rankn_conv
# lane ag-ho-core (H1, docs/plans/AUTOGRAD_HIGHER_ORDER.md): the higher-order bodies
from autograd.higher_ops import OP_HO_FIRST, OP_HO_LAST, apply_higher


@always_inline
def apply[OP: Int](t: Int, a: Args):
    # The autograd family (autograd/ops.mojo) is one contiguous range of
    # codes, one body each; no other operation code may fall inside it.
    comptime assert OP_AG_FIRST <= OP_AG_LAST, "autograd opcode range is empty"
    comptime assert (OP_CELL_FWD_SCAN < OP_AG_FIRST or OP_CELL_FWD_SCAN > OP_AG_LAST) and (
        OP_CELL_BWD_SCAN < OP_AG_FIRST or OP_CELL_BWD_SCAN > OP_AG_LAST
    ), "autograd opcode collides with recurrent scan"
    # The data-side autograd family (autograd/data_ops.mojo: dropout) is a
    # second contiguous range, disjoint from the first and from the scans.
    comptime assert OP_AGD_FIRST <= OP_AGD_LAST, "autograd data opcode range is empty"
    comptime assert OP_AGD_FIRST > OP_AG_LAST or OP_AGD_LAST < OP_AG_FIRST, "autograd opcode ranges overlap"
    comptime assert (OP_CELL_FWD_SCAN < OP_AGD_FIRST or OP_CELL_FWD_SCAN > OP_AGD_LAST) and (
        OP_CELL_BWD_SCAN < OP_AGD_FIRST or OP_CELL_BWD_SCAN > OP_AGD_LAST
    ), "autograd data opcode collides with recurrent scan"
    # The autograd optimizer family (autograd/optim.mojo), a range of its own.
    comptime assert OP_AGO_FIRST <= OP_AGO_LAST, "autograd optim opcode range is empty"
    comptime assert OP_AGO_LAST < OP_AG_FIRST or OP_AGO_FIRST > OP_AG_LAST, "autograd optim opcodes overlap the tape's"
    comptime assert (OP_CELL_FWD_SCAN < OP_AGO_FIRST or OP_CELL_FWD_SCAN > OP_AGO_LAST) and (
        OP_CELL_BWD_SCAN < OP_AGO_FIRST or OP_CELL_BWD_SCAN > OP_AGO_LAST
    ), "autograd optim opcode collides with recurrent scan"
    comptime assert OP_AGO_LAST < OP_AGD_FIRST or OP_AGO_FIRST > OP_AGD_LAST, "autograd optim opcodes overlap the data family"
    # The loss and metric family (autograd/loss_ops.mojo) is a second
    # contiguous range, disjoint from the autograd ops and the scan codes.
    comptime assert OP_LS_FIRST <= OP_LS_LAST, "loss opcode range is empty"
    comptime assert OP_LS_FIRST > OP_AG_LAST or OP_LS_LAST < OP_AG_FIRST, "loss opcodes collide with autograd ops"
    comptime assert (OP_CELL_FWD_SCAN < OP_LS_FIRST or OP_CELL_FWD_SCAN > OP_LS_LAST) and (
        OP_CELL_BWD_SCAN < OP_LS_FIRST or OP_CELL_BWD_SCAN > OP_LS_LAST
    ), "loss opcode collides with recurrent scan"
    comptime assert OP_LS_LAST < OP_AGD_FIRST or OP_LS_FIRST > OP_AGD_LAST, "loss opcodes overlap the data family"
    comptime assert OP_LS_LAST < OP_AGO_FIRST or OP_LS_FIRST > OP_AGO_LAST, "loss opcodes overlap the optim family"
    # The model trainer's family (autograd/model_ops.mojo) is a second range
    # that must not overlap the tape's or the scan codes.
    comptime assert OP_AGM_FIRST <= OP_AGM_LAST, "model opcode range is empty"
    comptime assert OP_AGM_FIRST > OP_AG_LAST or OP_AGM_LAST < OP_AG_FIRST, "model opcodes collide with autograd"
    comptime assert (OP_CELL_FWD_SCAN < OP_AGM_FIRST or OP_CELL_FWD_SCAN > OP_AGM_LAST) and (
        OP_CELL_BWD_SCAN < OP_AGM_FIRST or OP_CELL_BWD_SCAN > OP_AGM_LAST
    ), "model opcode collides with recurrent scan"
    comptime assert OP_AGM_LAST < OP_AGD_FIRST or OP_AGM_FIRST > OP_AGD_LAST, "model opcodes overlap the data family"
    comptime assert OP_AGM_LAST < OP_AGO_FIRST or OP_AGM_FIRST > OP_AGO_LAST, "model opcodes overlap the optim family"
    comptime assert OP_AGM_LAST < OP_LS_FIRST or OP_AGM_FIRST > OP_LS_LAST, "model opcodes overlap the loss family"
    # The tape's kernel family (lane ag-pass2-kernels, autograd/ops.mojo
    # OP_AGK_*, 1400..), disjoint from every other autograd range and the scans.
    comptime assert OP_AGK_FIRST <= OP_AGK_LAST, "autograd kernel opcode range is empty"
    comptime assert OP_AGK_FIRST > OP_AG_LAST or OP_AGK_LAST < OP_AG_FIRST, "autograd kernel opcodes overlap the tape's"
    comptime assert OP_AGK_FIRST > OP_AGD_LAST or OP_AGK_LAST < OP_AGD_FIRST, "autograd kernel opcodes overlap the data family"
    comptime assert OP_AGK_FIRST > OP_AGO_LAST or OP_AGK_LAST < OP_AGO_FIRST, "autograd kernel opcodes overlap the optim family"
    comptime assert OP_AGK_FIRST > OP_LS_LAST or OP_AGK_LAST < OP_LS_FIRST, "autograd kernel opcodes overlap the loss family"
    comptime assert OP_AGK_FIRST > OP_AGM_LAST or OP_AGK_LAST < OP_AGM_FIRST, "autograd kernel opcodes overlap the model family"
    comptime assert (OP_CELL_FWD_SCAN < OP_AGK_FIRST or OP_CELL_FWD_SCAN > OP_AGK_LAST) and (
        OP_CELL_BWD_SCAN < OP_AGK_FIRST or OP_CELL_BWD_SCAN > OP_AGK_LAST
    ), "autograd kernel opcode collides with recurrent scan"
    # The tape's fused nodes (autograd/pass1_ops.mojo), a range of their own.
    comptime assert OP_P1_FIRST <= OP_P1_LAST, "autograd fused opcode range is empty"
    comptime assert OP_P1_FIRST > OP_AG_LAST or OP_P1_LAST < OP_AG_FIRST, "fused opcodes collide with autograd"
    comptime assert (OP_CELL_FWD_SCAN < OP_P1_FIRST or OP_CELL_FWD_SCAN > OP_P1_LAST) and (
        OP_CELL_BWD_SCAN < OP_P1_FIRST or OP_CELL_BWD_SCAN > OP_P1_LAST
    ), "fused opcode collides with recurrent scan"
    comptime assert OP_P1_LAST < OP_AGD_FIRST or OP_P1_FIRST > OP_AGD_LAST, "fused opcodes overlap the data family"
    comptime assert OP_P1_LAST < OP_AGO_FIRST or OP_P1_FIRST > OP_AGO_LAST, "fused opcodes overlap the optim family"
    comptime assert OP_P1_LAST < OP_LS_FIRST or OP_P1_FIRST > OP_LS_LAST, "fused opcodes overlap the loss family"
    comptime assert OP_P1_LAST < OP_AGM_FIRST or OP_P1_FIRST > OP_AGM_LAST, "fused opcodes overlap the model family"
    comptime assert OP_P1_LAST < OP_AGK_FIRST or OP_P1_FIRST > OP_AGK_LAST, "fused opcodes overlap the kernel family"
    # The MLP loss block (sequence/mlp.mojo MLP_LOSSBLOCK, lane
    # ag-pass3-lossblock, 1600..), disjoint from every autograd range and the scans.
    comptime assert OP_MLP_LB_FIRST <= OP_MLP_LB_LAST, "MLP loss-block opcode range is empty"
    comptime assert OP_MLP_LB_FIRST > OP_AG_LAST or OP_MLP_LB_LAST < OP_AG_FIRST, "MLP loss-block opcodes overlap the tape's"
    comptime assert OP_MLP_LB_FIRST > OP_AGD_LAST or OP_MLP_LB_LAST < OP_AGD_FIRST, "MLP loss-block opcodes overlap the data family"
    comptime assert OP_MLP_LB_FIRST > OP_AGO_LAST or OP_MLP_LB_LAST < OP_AGO_FIRST, "MLP loss-block opcodes overlap the optim family"
    comptime assert OP_MLP_LB_FIRST > OP_LS_LAST or OP_MLP_LB_LAST < OP_LS_FIRST, "MLP loss-block opcodes overlap the loss family"
    comptime assert OP_MLP_LB_FIRST > OP_AGM_LAST or OP_MLP_LB_LAST < OP_AGM_FIRST, "MLP loss-block opcodes overlap the model family"
    comptime assert OP_MLP_LB_FIRST > OP_AGK_LAST or OP_MLP_LB_LAST < OP_AGK_FIRST, "MLP loss-block opcodes overlap the kernel family"
    comptime assert OP_MLP_LB_FIRST > OP_P1_LAST or OP_MLP_LB_LAST < OP_P1_FIRST, "MLP loss-block opcodes overlap the fused family"
    comptime assert (OP_CELL_FWD_SCAN < OP_MLP_LB_FIRST or OP_CELL_FWD_SCAN > OP_MLP_LB_LAST) and (
        OP_CELL_BWD_SCAN < OP_MLP_LB_FIRST or OP_CELL_BWD_SCAN > OP_MLP_LB_LAST
    ), "MLP loss-block opcode collides with recurrent scan"
    # The fold tree's level family (autograd/ops.mojo OP_AGF_*, 1700..; lane
    # ag-pass3-fold), disjoint from every other autograd range and the scans.
    comptime assert OP_AGF_FIRST <= OP_AGF_LAST, "fold level opcode range is empty"
    comptime assert OP_AGF_FIRST > OP_AG_LAST or OP_AGF_LAST < OP_AG_FIRST, "fold level opcodes overlap the tape's"
    comptime assert OP_AGF_FIRST > OP_AGD_LAST or OP_AGF_LAST < OP_AGD_FIRST, "fold level opcodes overlap the data family"
    comptime assert OP_AGF_FIRST > OP_AGO_LAST or OP_AGF_LAST < OP_AGO_FIRST, "fold level opcodes overlap the optim family"
    comptime assert OP_AGF_FIRST > OP_LS_LAST or OP_AGF_LAST < OP_LS_FIRST, "fold level opcodes overlap the loss family"
    comptime assert OP_AGF_FIRST > OP_AGM_LAST or OP_AGF_LAST < OP_AGM_FIRST, "fold level opcodes overlap the model family"
    comptime assert OP_AGF_FIRST > OP_AGK_LAST or OP_AGF_LAST < OP_AGK_FIRST, "fold level opcodes overlap the kernel family"
    comptime assert OP_AGF_FIRST > OP_P1_LAST or OP_AGF_LAST < OP_P1_FIRST, "fold level opcodes overlap the fused family"
    comptime assert (OP_CELL_FWD_SCAN < OP_AGF_FIRST or OP_CELL_FWD_SCAN > OP_AGF_LAST) and (
        OP_CELL_BWD_SCAN < OP_AGF_FIRST or OP_CELL_BWD_SCAN > OP_AGF_LAST
    ), "fold level opcode collides with recurrent scan"
    comptime assert OP_AGF_FIRST > OP_MLP_LB_LAST or OP_AGF_LAST < OP_MLP_LB_FIRST, "fold level opcodes overlap the MLP loss block"
    # The composed-step family (autograd/composed_ops.mojo OP_AGC_*, 2000..;
    # lane ag-pass5-composed), disjoint from every other autograd range and the scans.
    comptime assert OP_AGC_FIRST <= OP_AGC_LAST, "composed-step opcode range is empty"
    comptime assert OP_AGC_FIRST > OP_AG_LAST or OP_AGC_LAST < OP_AG_FIRST, "composed-step opcodes overlap the tape's"
    comptime assert OP_AGC_FIRST > OP_AGD_LAST or OP_AGC_LAST < OP_AGD_FIRST, "composed-step opcodes overlap the data family"
    comptime assert OP_AGC_FIRST > OP_AGO_LAST or OP_AGC_LAST < OP_AGO_FIRST, "composed-step opcodes overlap the optim family"
    comptime assert OP_AGC_FIRST > OP_LS_LAST or OP_AGC_LAST < OP_LS_FIRST, "composed-step opcodes overlap the loss family"
    comptime assert OP_AGC_FIRST > OP_AGM_LAST or OP_AGC_LAST < OP_AGM_FIRST, "composed-step opcodes overlap the model family"
    comptime assert OP_AGC_FIRST > OP_AGK_LAST or OP_AGC_LAST < OP_AGK_FIRST, "composed-step opcodes overlap the kernel family"
    comptime assert OP_AGC_FIRST > OP_P1_LAST or OP_AGC_LAST < OP_P1_FIRST, "composed-step opcodes overlap the fused family"
    comptime assert OP_AGC_FIRST > OP_MLP_LB_LAST or OP_AGC_LAST < OP_MLP_LB_FIRST, "composed-step opcodes overlap the MLP loss block"
    comptime assert OP_AGC_FIRST > OP_AGF_LAST or OP_AGC_LAST < OP_AGF_FIRST, "composed-step opcodes overlap the fold levels"
    comptime assert (OP_CELL_FWD_SCAN < OP_AGC_FIRST or OP_CELL_FWD_SCAN > OP_AGC_LAST) and (
        OP_CELL_BWD_SCAN < OP_AGC_FIRST or OP_CELL_BWD_SCAN > OP_AGC_LAST
    ), "composed-step opcode collides with recurrent scan"
    # The pass-6 rows family (autograd/row_ops.mojo OP_ROW_*, 2200..; lane
    # ag-pass6-rows), disjoint from every other autograd range and the scans.
    comptime assert OP_ROW_FIRST <= OP_ROW_LAST, "rows opcode range is empty"
    comptime assert OP_ROW_FIRST > OP_AG_LAST or OP_ROW_LAST < OP_AG_FIRST, "rows opcodes overlap the tape's"
    comptime assert OP_ROW_FIRST > OP_AGD_LAST or OP_ROW_LAST < OP_AGD_FIRST, "rows opcodes overlap the data family"
    comptime assert OP_ROW_FIRST > OP_AGO_LAST or OP_ROW_LAST < OP_AGO_FIRST, "rows opcodes overlap the optim family"
    comptime assert OP_ROW_FIRST > OP_LS_LAST or OP_ROW_LAST < OP_LS_FIRST, "rows opcodes overlap the loss family"
    comptime assert OP_ROW_FIRST > OP_AGM_LAST or OP_ROW_LAST < OP_AGM_FIRST, "rows opcodes overlap the model family"
    comptime assert OP_ROW_FIRST > OP_AGK_LAST or OP_ROW_LAST < OP_AGK_FIRST, "rows opcodes overlap the kernel family"
    comptime assert OP_ROW_FIRST > OP_P1_LAST or OP_ROW_LAST < OP_P1_FIRST, "rows opcodes overlap the fused family"
    comptime assert OP_ROW_FIRST > OP_MLP_LB_LAST or OP_ROW_LAST < OP_MLP_LB_FIRST, "rows opcodes overlap the MLP loss block"
    comptime assert OP_ROW_FIRST > OP_AGF_LAST or OP_ROW_LAST < OP_AGF_FIRST, "rows opcodes overlap the fold levels"
    comptime assert OP_ROW_FIRST > OP_AGC_LAST or OP_ROW_LAST < OP_AGC_FIRST, "rows opcodes overlap the composed step"
    comptime assert (OP_CELL_FWD_SCAN < OP_ROW_FIRST or OP_CELL_FWD_SCAN > OP_ROW_LAST) and (
        OP_CELL_BWD_SCAN < OP_ROW_FIRST or OP_CELL_BWD_SCAN > OP_ROW_LAST
    ), "rows opcode collides with recurrent scan"
    # The pass-6 host lane's family (autograd/optim.mojo OP_AGH_*, 2400..;
    # lane ag-pass6-host), disjoint from every other autograd range and the scans.
    comptime assert OP_AGH_FIRST <= OP_AGH_LAST, "pass-6 host opcode range is empty"
    comptime assert OP_AGH_FIRST > OP_AG_LAST or OP_AGH_LAST < OP_AG_FIRST, "pass-6 host opcodes overlap the tape's"
    comptime assert OP_AGH_FIRST > OP_AGD_LAST or OP_AGH_LAST < OP_AGD_FIRST, "pass-6 host opcodes overlap the data family"
    comptime assert OP_AGH_FIRST > OP_AGO_LAST or OP_AGH_LAST < OP_AGO_FIRST, "pass-6 host opcodes overlap the optim family"
    comptime assert OP_AGH_FIRST > OP_LS_LAST or OP_AGH_LAST < OP_LS_FIRST, "pass-6 host opcodes overlap the loss family"
    comptime assert OP_AGH_FIRST > OP_AGM_LAST or OP_AGH_LAST < OP_AGM_FIRST, "pass-6 host opcodes overlap the model family"
    comptime assert OP_AGH_FIRST > OP_AGK_LAST or OP_AGH_LAST < OP_AGK_FIRST, "pass-6 host opcodes overlap the kernel family"
    comptime assert OP_AGH_FIRST > OP_P1_LAST or OP_AGH_LAST < OP_P1_FIRST, "pass-6 host opcodes overlap the fused family"
    comptime assert OP_AGH_FIRST > OP_MLP_LB_LAST or OP_AGH_LAST < OP_MLP_LB_FIRST, "pass-6 host opcodes overlap the MLP loss block"
    comptime assert OP_AGH_FIRST > OP_AGF_LAST or OP_AGH_LAST < OP_AGF_FIRST, "pass-6 host opcodes overlap the fold levels"
    comptime assert OP_AGH_FIRST > OP_AGC_LAST or OP_AGH_LAST < OP_AGC_FIRST, "pass-6 host opcodes overlap the composed step"
    comptime assert (OP_CELL_FWD_SCAN < OP_AGH_FIRST or OP_CELL_FWD_SCAN > OP_AGH_LAST) and (
        OP_CELL_BWD_SCAN < OP_AGH_FIRST or OP_CELL_BWD_SCAN > OP_AGH_LAST
    ), "pass-6 host opcode collides with recurrent scan"
    comptime assert OP_AGH_FIRST > OP_ROW_LAST or OP_AGH_LAST < OP_ROW_FIRST, "pass-6 host opcodes overlap the rows family"
    # The fold lane's loss-leaf family (autograd/fold_ops.mojo OP_AGS_*, 2300..;
    # lane ag-pass6-fold), disjoint from every other autograd range and the scans.
    comptime assert OP_AGS_FIRST <= OP_AGS_LAST, "loss-leaf opcode range is empty"
    comptime assert OP_AGS_FIRST > OP_AG_LAST or OP_AGS_LAST < OP_AG_FIRST, "loss-leaf opcodes overlap the tape's"
    comptime assert OP_AGS_FIRST > OP_AGD_LAST or OP_AGS_LAST < OP_AGD_FIRST, "loss-leaf opcodes overlap the data family"
    comptime assert OP_AGS_FIRST > OP_AGO_LAST or OP_AGS_LAST < OP_AGO_FIRST, "loss-leaf opcodes overlap the optim family"
    comptime assert OP_AGS_FIRST > OP_LS_LAST or OP_AGS_LAST < OP_LS_FIRST, "loss-leaf opcodes overlap the loss family"
    comptime assert OP_AGS_FIRST > OP_AGM_LAST or OP_AGS_LAST < OP_AGM_FIRST, "loss-leaf opcodes overlap the model family"
    comptime assert OP_AGS_FIRST > OP_AGK_LAST or OP_AGS_LAST < OP_AGK_FIRST, "loss-leaf opcodes overlap the kernel family"
    comptime assert OP_AGS_FIRST > OP_P1_LAST or OP_AGS_LAST < OP_P1_FIRST, "loss-leaf opcodes overlap the fused family"
    comptime assert OP_AGS_FIRST > OP_MLP_LB_LAST or OP_AGS_LAST < OP_MLP_LB_FIRST, "loss-leaf opcodes overlap the MLP loss block"
    comptime assert OP_AGS_FIRST > OP_AGF_LAST or OP_AGS_LAST < OP_AGF_FIRST, "loss-leaf opcodes overlap the fold levels"
    comptime assert OP_AGS_FIRST > OP_AGC_LAST or OP_AGS_LAST < OP_AGC_FIRST, "loss-leaf opcodes overlap the composed step"
    comptime assert (OP_CELL_FWD_SCAN < OP_AGS_FIRST or OP_CELL_FWD_SCAN > OP_AGS_LAST) and (
        OP_CELL_BWD_SCAN < OP_AGS_FIRST or OP_CELL_BWD_SCAN > OP_AGS_LAST
    ), "loss-leaf opcode collides with recurrent scan"
    # The pass-6 GEMM family (sequence/gemm_split.mojo OP_GK_*, 2100..;
    # lane ag-pass6-gemm), disjoint from every autograd range and the scans.
    comptime assert OP_GK_FIRST <= OP_GK_LAST, "pass-6 gemm opcode range is empty"
    comptime assert OP_GK_FIRST > OP_AGC_LAST or OP_GK_LAST < OP_AGC_FIRST, "pass-6 gemm opcodes overlap the composed family"
    comptime assert OP_GK_FIRST > OP_AG_LAST or OP_GK_LAST < OP_AG_FIRST, "pass-6 gemm opcodes overlap the tape's"
    comptime assert OP_GK_FIRST > OP_AGK_LAST or OP_GK_LAST < OP_AGK_FIRST, "pass-6 gemm opcodes overlap the kernel family"
    comptime assert OP_GK_FIRST > OP_AGF_LAST or OP_GK_LAST < OP_AGF_FIRST, "pass-6 gemm opcodes overlap the fold levels"
    comptime assert OP_GK_FIRST > OP_P1_LAST or OP_GK_LAST < OP_P1_FIRST, "pass-6 gemm opcodes overlap the fused family"
    comptime assert OP_GK_FIRST > OP_MLP_LB_LAST or OP_GK_LAST < OP_MLP_LB_FIRST, "pass-6 gemm opcodes overlap the MLP loss block"
    comptime assert (OP_CELL_FWD_SCAN < OP_GK_FIRST or OP_CELL_FWD_SCAN > OP_GK_LAST) and (
        OP_CELL_BWD_SCAN < OP_GK_FIRST or OP_CELL_BWD_SCAN > OP_GK_LAST
    ), "pass-6 gemm opcode collides with recurrent scan"
    # The MoE backward family (autograd/kernels/moe.mojo OP_MOEB_*, 2900..;
    # lane ag-moe-backward), disjoint from every autograd range and the scans.
    comptime assert OP_MOEB_FIRST <= OP_MOEB_LAST, "MoE backward opcode range is empty"
    comptime assert OP_MOEB_FIRST > OP_AG_LAST or OP_MOEB_LAST < OP_AG_FIRST, "MoE backward opcodes overlap the tape's"
    comptime assert OP_MOEB_FIRST > OP_AGD_LAST or OP_MOEB_LAST < OP_AGD_FIRST, "MoE backward opcodes overlap the data family"
    comptime assert OP_MOEB_FIRST > OP_AGO_LAST or OP_MOEB_LAST < OP_AGO_FIRST, "MoE backward opcodes overlap the optim family"
    comptime assert OP_MOEB_FIRST > OP_LS_LAST or OP_MOEB_LAST < OP_LS_FIRST, "MoE backward opcodes overlap the loss family"
    comptime assert OP_MOEB_FIRST > OP_AGM_LAST or OP_MOEB_LAST < OP_AGM_FIRST, "MoE backward opcodes overlap the model family"
    comptime assert OP_MOEB_FIRST > OP_AGK_LAST or OP_MOEB_LAST < OP_AGK_FIRST, "MoE backward opcodes overlap the kernel family"
    comptime assert OP_MOEB_FIRST > OP_P1_LAST or OP_MOEB_LAST < OP_P1_FIRST, "MoE backward opcodes overlap the fused family"
    comptime assert OP_MOEB_FIRST > OP_MLP_LB_LAST or OP_MOEB_LAST < OP_MLP_LB_FIRST, "MoE backward opcodes overlap the MLP loss block"
    comptime assert OP_MOEB_FIRST > OP_AGF_LAST or OP_MOEB_LAST < OP_AGF_FIRST, "MoE backward opcodes overlap the fold levels"
    comptime assert OP_MOEB_FIRST > OP_AGC_LAST or OP_MOEB_LAST < OP_AGC_FIRST, "MoE backward opcodes overlap the composed step"
    comptime assert OP_MOEB_FIRST > OP_ROW_LAST or OP_MOEB_LAST < OP_ROW_FIRST, "MoE backward opcodes overlap the rows family"
    comptime assert OP_MOEB_FIRST > OP_AGH_LAST or OP_MOEB_LAST < OP_AGH_FIRST, "MoE backward opcodes overlap the pass-6 host family"
    comptime assert OP_MOEB_FIRST > OP_AGS_LAST or OP_MOEB_LAST < OP_AGS_FIRST, "MoE backward opcodes overlap the loss-leaf family"
    comptime assert OP_MOEB_FIRST > OP_GK_LAST or OP_MOEB_LAST < OP_GK_FIRST, "MoE backward opcodes overlap the pass-6 gemm family"
    comptime assert (OP_CELL_FWD_SCAN < OP_MOEB_FIRST or OP_CELL_FWD_SCAN > OP_MOEB_LAST) and (
        OP_CELL_BWD_SCAN < OP_MOEB_FIRST or OP_CELL_BWD_SCAN > OP_MOEB_LAST
    ), "MoE backward opcode collides with recurrent scan"
    # The embedding tape op's family (autograd/embedding_ops.mojo OP_EMB_*,
    # 2700..; lane ag-embedding-op), above every other autograd range.
    comptime assert OP_EMB_FIRST <= OP_EMB_LAST, "embedding opcode range is empty"
    comptime assert OP_EMB_FIRST > OP_AGH_LAST and OP_EMB_FIRST > OP_AGS_LAST and OP_EMB_FIRST > OP_ROW_LAST and OP_EMB_FIRST > OP_GK_LAST and OP_EMB_FIRST > OP_AGC_LAST, "embedding opcodes overlap a lower family"
    comptime assert OP_EMB_FIRST > OP_AG_LAST and OP_EMB_FIRST > OP_AGD_LAST and OP_EMB_FIRST > OP_AGO_LAST and OP_EMB_FIRST > OP_LS_LAST and OP_EMB_FIRST > OP_AGM_LAST and OP_EMB_FIRST > OP_AGK_LAST and OP_EMB_FIRST > OP_P1_LAST and OP_EMB_FIRST > OP_AGF_LAST and OP_EMB_FIRST > OP_MLP_LB_LAST, "embedding opcodes overlap an autograd family"
    comptime assert (OP_CELL_FWD_SCAN < OP_EMB_FIRST or OP_CELL_FWD_SCAN > OP_EMB_LAST) and (
        OP_CELL_BWD_SCAN < OP_EMB_FIRST or OP_CELL_BWD_SCAN > OP_EMB_LAST
    ), "embedding opcode collides with recurrent scan"
    # The recorded loss VJPs' constants (autograd/higher_loss_ops.mojo
    # OP_HL_*, 3610..3619; lane ag-ho-fused, H2 of AUTOGRAD_HIGHER_ORDER.md),
    # above the cast range (itself above every other autograd range); the
    # fence against H1's OP_HO_* range (3600..3609) is in apply_higher_loss.
    comptime assert OP_HL_FIRST <= OP_HL_LAST, "recorded-loss opcode range is empty"
    comptime assert OP_HL_FIRST > OP_CAST_LAST and OP_HL_FIRST > OP_EMB_LAST, "recorded-loss opcodes overlap a lower family"
    comptime assert (OP_CELL_FWD_SCAN < OP_HL_FIRST or OP_CELL_FWD_SCAN > OP_HL_LAST) and (
        OP_CELL_BWD_SCAN < OP_HL_FIRST or OP_CELL_BWD_SCAN > OP_HL_LAST
    ), "recorded-loss opcode collides with recurrent scan"
    # The input cast and label gather (autograd/cast_ops.mojo OP_CAST_*,
    # 3500..; lane ag-thin-glue), above every other autograd range.
    comptime assert OP_CAST_FIRST <= OP_CAST_LAST, "cast opcode range is empty"
    comptime assert OP_CAST_FIRST > OP_EMB_LAST and OP_CAST_FIRST > OP_MOEB_LAST and OP_CAST_FIRST > OP_AGH_LAST and OP_CAST_FIRST > OP_AGS_LAST and OP_CAST_FIRST > OP_ROW_LAST and OP_CAST_FIRST > OP_GK_LAST and OP_CAST_FIRST > OP_AGC_LAST, "cast opcodes overlap a lower family"
    comptime assert OP_CAST_FIRST > OP_AG_LAST and OP_CAST_FIRST > OP_AGD_LAST and OP_CAST_FIRST > OP_AGO_LAST and OP_CAST_FIRST > OP_LS_LAST and OP_CAST_FIRST > OP_AGM_LAST and OP_CAST_FIRST > OP_AGK_LAST and OP_CAST_FIRST > OP_P1_LAST and OP_CAST_FIRST > OP_AGF_LAST and OP_CAST_FIRST > OP_MLP_LB_LAST, "cast opcodes overlap an autograd family"
    comptime assert (OP_CELL_FWD_SCAN < OP_CAST_FIRST or OP_CELL_FWD_SCAN > OP_CAST_LAST) and (
        OP_CELL_BWD_SCAN < OP_CAST_FIRST or OP_CELL_BWD_SCAN > OP_CAST_LAST
    ), "cast opcode collides with recurrent scan"
    # ---- lane ag-rankn-core: the rank-N view copy (autograd/rankn_ops.mojo
    # OP_RNC_*, 4100..4109; docs/plans/AUTOGRAD_RANKN.md section 7), above the
    # cast family and the higher-order plan's 3600..3619.
    comptime assert OP_RNC_FIRST <= OP_RNC_LAST, "rank-N core opcode range is empty"
    comptime assert OP_RNC_FIRST > OP_CAST_LAST and OP_RNC_FIRST > OP_EMB_LAST and OP_RNC_FIRST > OP_MOEB_LAST and OP_RNC_FIRST > OP_AGH_LAST and OP_RNC_FIRST > OP_AGS_LAST and OP_RNC_FIRST > OP_ROW_LAST and OP_RNC_FIRST > OP_GK_LAST and OP_RNC_FIRST > OP_AGC_LAST, "rank-N core opcodes overlap a lower family"
    comptime assert OP_RNC_FIRST > OP_AG_LAST and OP_RNC_FIRST > OP_AGD_LAST and OP_RNC_FIRST > OP_AGO_LAST and OP_RNC_FIRST > OP_LS_LAST and OP_RNC_FIRST > OP_AGM_LAST and OP_RNC_FIRST > OP_AGK_LAST and OP_RNC_FIRST > OP_P1_LAST and OP_RNC_FIRST > OP_AGF_LAST and OP_RNC_FIRST > OP_MLP_LB_LAST, "rank-N core opcodes overlap an autograd family"
    comptime assert OP_RNC_FIRST > 3619, "rank-N core opcodes overlap the higher-order range 3600..3619"
    comptime assert (OP_CELL_FWD_SCAN < OP_RNC_FIRST or OP_CELL_FWD_SCAN > OP_RNC_LAST) and (
        OP_CELL_BWD_SCAN < OP_RNC_FIRST or OP_CELL_BWD_SCAN > OP_RNC_LAST
    ), "rank-N core opcode collides with recurrent scan"
    # ---- end lane ag-rankn-core
    # >>> lane ag-rankn-fold: the rank-N broadcast / reduction / max family
    # (autograd/rankn_fold_ops.mojo OP_RNF_*, 4200..4219; docs/plans/AUTOGRAD_RANKN.md 7)
    comptime assert OP_RNF_FIRST <= OP_RNF_LAST, "rank-N fold opcode range is empty"
    comptime assert OP_RNF_FIRST > OP_CAST_LAST and OP_RNF_FIRST > OP_EMB_LAST and OP_RNF_FIRST > OP_MOEB_LAST and OP_RNF_FIRST > OP_AGH_LAST and OP_RNF_FIRST > OP_AGS_LAST and OP_RNF_FIRST > OP_ROW_LAST and OP_RNF_FIRST > OP_GK_LAST and OP_RNF_FIRST > OP_AGC_LAST, "rank-N fold opcodes overlap a lower family"
    comptime assert OP_RNF_FIRST > OP_AG_LAST and OP_RNF_FIRST > OP_AGD_LAST and OP_RNF_FIRST > OP_AGO_LAST and OP_RNF_FIRST > OP_LS_LAST and OP_RNF_FIRST > OP_AGM_LAST and OP_RNF_FIRST > OP_AGK_LAST and OP_RNF_FIRST > OP_P1_LAST and OP_RNF_FIRST > OP_AGF_LAST and OP_RNF_FIRST > OP_MLP_LB_LAST, "rank-N fold opcodes overlap an autograd family"
    comptime assert (OP_CELL_FWD_SCAN < OP_RNF_FIRST or OP_CELL_FWD_SCAN > OP_RNF_LAST) and (
        OP_CELL_BWD_SCAN < OP_RNF_FIRST or OP_CELL_BWD_SCAN > OP_RNF_LAST
    ), "rank-N fold opcode collides with recurrent scan"
    # <<< lane ag-rankn-fold
    # lane ag-rankn-bmm: the batched matrix product (autograd/rankn_bmm_ops.mojo
    # OP_RNB_*, 4300..4309), above every other autograd range.
    comptime assert OP_RNB_FIRST <= OP_RNB_LAST, "rank-N bmm opcode range is empty"
    comptime assert OP_RNB_FIRST > OP_CAST_LAST and OP_RNB_FIRST > OP_EMB_LAST and OP_RNB_FIRST > OP_MOEB_LAST and OP_RNB_FIRST > OP_AGH_LAST and OP_RNB_FIRST > OP_AGS_LAST and OP_RNB_FIRST > OP_ROW_LAST and OP_RNB_FIRST > OP_GK_LAST and OP_RNB_FIRST > OP_AGC_LAST, "rank-N bmm opcodes overlap a lower family"
    comptime assert OP_RNB_FIRST > OP_AG_LAST and OP_RNB_FIRST > OP_AGD_LAST and OP_RNB_FIRST > OP_AGO_LAST and OP_RNB_FIRST > OP_LS_LAST and OP_RNB_FIRST > OP_AGM_LAST and OP_RNB_FIRST > OP_AGK_LAST and OP_RNB_FIRST > OP_P1_LAST and OP_RNB_FIRST > OP_AGF_LAST and OP_RNB_FIRST > OP_MLP_LB_LAST, "rank-N bmm opcodes overlap an autograd family"
    comptime assert (OP_CELL_FWD_SCAN < OP_RNB_FIRST or OP_CELL_FWD_SCAN > OP_RNB_LAST) and (
        OP_CELL_BWD_SCAN < OP_RNB_FIRST or OP_CELL_BWD_SCAN > OP_RNB_LAST
    ), "rank-N bmm opcode collides with recurrent scan"
    # ---- rank-N layout (lane ag-rankn-layout; autograd/rankn_layout_ops.mojo
    # OP_RNL_*, 4400..4409, docs/plans/AUTOGRAD_RANKN.md section 7) ----
    comptime assert OP_RNL_FIRST <= OP_RNL_LAST, "rank-N layout opcode range is empty"
    comptime assert OP_RNL_FIRST > OP_CAST_LAST and OP_RNL_FIRST > OP_EMB_LAST and OP_RNL_FIRST > OP_MOEB_LAST and OP_RNL_FIRST > OP_AGH_LAST and OP_RNL_FIRST > OP_AGS_LAST and OP_RNL_FIRST > OP_ROW_LAST and OP_RNL_FIRST > OP_GK_LAST and OP_RNL_FIRST > OP_AGC_LAST, "rank-N layout opcodes overlap a lower family"
    comptime assert OP_RNL_FIRST > OP_AG_LAST and OP_RNL_FIRST > OP_AGD_LAST and OP_RNL_FIRST > OP_AGO_LAST and OP_RNL_FIRST > OP_LS_LAST and OP_RNL_FIRST > OP_AGM_LAST and OP_RNL_FIRST > OP_AGK_LAST and OP_RNL_FIRST > OP_P1_LAST and OP_RNL_FIRST > OP_AGF_LAST and OP_RNL_FIRST > OP_MLP_LB_LAST, "rank-N layout opcodes overlap an autograd family"
    comptime assert (OP_CELL_FWD_SCAN < OP_RNL_FIRST or OP_CELL_FWD_SCAN > OP_RNL_LAST) and (
        OP_CELL_BWD_SCAN < OP_RNL_FIRST or OP_CELL_BWD_SCAN > OP_RNL_LAST
    ), "rank-N layout opcode collides with recurrent scan"
    # ---- end rank-N layout ----
    # ---- lane ag-rankn-conv: the rank-4 conv2d / pool family (autograd/
    # rankn_conv_ops.mojo OP_RNV_*, 4500..4519), above every lower family.
    comptime assert OP_RNV_FIRST <= OP_RNV_LAST, "rank-N conv opcode range is empty"
    comptime assert OP_RNV_FIRST > OP_CAST_LAST and OP_RNV_FIRST > OP_EMB_LAST and OP_RNV_FIRST > OP_MOEB_LAST and OP_RNV_FIRST > OP_AGH_LAST and OP_RNV_FIRST > OP_AGS_LAST and OP_RNV_FIRST > OP_ROW_LAST and OP_RNV_FIRST > OP_GK_LAST and OP_RNV_FIRST > OP_AGC_LAST, "rank-N conv opcodes overlap a lower family"
    comptime assert OP_RNV_FIRST > OP_AG_LAST and OP_RNV_FIRST > OP_AGD_LAST and OP_RNV_FIRST > OP_AGO_LAST and OP_RNV_FIRST > OP_LS_LAST and OP_RNV_FIRST > OP_AGM_LAST and OP_RNV_FIRST > OP_AGK_LAST and OP_RNV_FIRST > OP_P1_LAST and OP_RNV_FIRST > OP_AGF_LAST and OP_RNV_FIRST > OP_MLP_LB_LAST, "rank-N conv opcodes overlap an autograd family"
    comptime assert (OP_CELL_FWD_SCAN < OP_RNV_FIRST or OP_CELL_FWD_SCAN > OP_RNV_LAST) and (
        OP_CELL_BWD_SCAN < OP_RNV_FIRST or OP_CELL_BWD_SCAN > OP_RNV_LAST
    ), "rank-N conv opcode collides with recurrent scan"
    # ---- end lane ag-rankn-conv
    # lane ag-ho-core (H1): the higher-order bodies (autograd/higher_ops.mojo
    # OP_HO_*, 3600..3609), above the cast family.
    comptime assert OP_HO_FIRST <= OP_HO_LAST, "higher-order opcode range is empty"
    comptime assert OP_HO_FIRST > OP_CAST_LAST and OP_HO_FIRST > OP_EMB_LAST and OP_HO_FIRST > OP_MOEB_LAST and OP_HO_FIRST > OP_AGH_LAST and OP_HO_FIRST > OP_AGS_LAST and OP_HO_FIRST > OP_ROW_LAST and OP_HO_FIRST > OP_GK_LAST and OP_HO_FIRST > OP_AGC_LAST, "higher-order opcodes overlap a lower family"
    comptime assert OP_HO_FIRST > OP_AG_LAST and OP_HO_FIRST > OP_AGD_LAST and OP_HO_FIRST > OP_AGO_LAST and OP_HO_FIRST > OP_LS_LAST and OP_HO_FIRST > OP_AGM_LAST and OP_HO_FIRST > OP_AGK_LAST and OP_HO_FIRST > OP_P1_LAST and OP_HO_FIRST > OP_AGF_LAST and OP_HO_FIRST > OP_MLP_LB_LAST, "higher-order opcodes overlap an autograd family"
    comptime assert (OP_CELL_FWD_SCAN < OP_HO_FIRST or OP_CELL_FWD_SCAN > OP_HO_LAST) and (
        OP_CELL_BWD_SCAN < OP_HO_FIRST or OP_CELL_BWD_SCAN > OP_HO_LAST
    ), "higher-order opcode collides with recurrent scan"
    # MOJOLEARN_IDN_ROW_COOP (autograd/row_ops.mojo): the softmax rows in the
    # fold32 order on the host and the device's one-thread rows, the twin of
    # the coop cell DeviceExec runs for rows of >= 32 columns
    comptime if ROW_COOP and OP == OP_AG_SOFTMAX_FWD:
        row_softmax_fwd(t, a)
    elif ROW_COOP and OP == OP_AG_SOFTMAX_VJP:
        row_softmax_vjp(t, a)
    elif OP >= OP_AG_FIRST and OP <= OP_AG_LAST:
        apply_autograd[OP](t, a)
    elif OP >= OP_AGD_FIRST and OP <= OP_AGD_LAST:
        apply_autograd_data[OP](t, a)
    elif OP >= OP_AGO_FIRST and OP <= OP_AGO_LAST:
        apply_optim[OP](t, a)
    elif OP >= OP_LS_FIRST and OP <= OP_LS_LAST:
        apply_losses[OP](t, a)
    elif OP >= OP_AGM_FIRST and OP <= OP_AGM_LAST:
        apply_model_ops[OP](t, a)
    elif OP == OP_AGK_GEMM_EPI:
        # the tape's GEMM with epilogue: op_gemm_epi's words with the follower
        # applied before the one store (lane ag-pass3-impl; DeviceExec tiles
        # it where OP_GEMM is tiled, sequence/gemm_tiled.mojo)
        op_agk_gemm_epi(t, a)
    elif OP >= OP_AGK_FIRST and OP <= OP_AGK_LAST:
        apply_autograd_kernels[OP](t, a)
    elif OP >= OP_P1_FIRST and OP <= OP_P1_LAST:
        apply_pass1[OP](t, a)
    elif OP >= OP_AGF_FIRST and OP <= OP_AGF_LAST:
        apply_autograd_fold[OP](t, a)
    elif OP >= OP_AGC_FIRST and OP <= OP_AGC_LAST:
        apply_composed[OP](t, a)
    elif OP >= OP_ROW_FIRST and OP <= OP_ROW_LAST:
        apply_rows[OP](t, a)
    elif OP >= OP_AGH_FIRST and OP <= OP_AGH_LAST:
        apply_optim_fused[OP](t, a)
    elif OP >= OP_AGS_FIRST and OP <= OP_AGS_LAST:
        apply_fold_ops[OP](t, a)
    elif OP >= OP_MOEB_FIRST and OP <= OP_MOEB_LAST:
        apply_moe_backward[OP](t, a)
    elif OP >= OP_EMB_FIRST and OP <= OP_EMB_LAST:
        apply_embedding[OP](t, a)
    elif OP >= OP_HL_FIRST and OP <= OP_HL_LAST:
        apply_higher_loss[OP](t, a)
    elif OP >= OP_CAST_FIRST and OP <= OP_CAST_LAST:
        apply_cast[OP](t, a)
    elif OP >= OP_RNC_FIRST and OP <= OP_RNC_LAST:
        # lane ag-rankn-core: the rank-N view copy
        apply_rankn_core[OP](t, a)
    # >>> lane ag-rankn-fold
    elif OP >= OP_RNF_FIRST and OP <= OP_RNF_LAST:
        apply_rankn_fold[OP](t, a)
    # <<< lane ag-rankn-fold
    elif OP >= OP_RNB_FIRST and OP <= OP_RNB_LAST:
        # lane ag-rankn-bmm: the batched matrix product
        apply_rn_bmm[OP](t, a)
    elif OP >= OP_RNL_FIRST and OP <= OP_RNL_LAST:
        # rank-N layout (lane ag-rankn-layout)
        apply_rankn_layout[OP](t, a)
    elif OP >= OP_RNV_FIRST and OP <= OP_RNV_LAST:
        # lane ag-rankn-conv: rank-4 conv2d and pools
        apply_rankn_conv[OP](t, a)
    elif OP >= OP_HO_FIRST and OP <= OP_HO_LAST:
        apply_higher[OP](t, a)
    elif OP == OP_GEMM:
        op_gemm(t, a)
    elif OP == OP_GEMM_SPLITK:
        op_gemm_splitk(t, a)
    elif OP == OP_GK_SPLIT:
        # lane ag-pass6-gemm: the split-K order of the three GEMM opcodes
        op_gk_split(t, a)
    elif OP == OP_BIAS:
        op_bias(t, a)
    elif OP == OP_COLSUM:
        op_colsum(t, a)
    elif OP == OP_CELL_FWD:
        op_cell_fwd(t, a)
    elif OP == OP_CELL_BWD:
        op_cell_bwd(t, a)
    elif OP == OP_CELL_FWD_H:
        op_cell_fwd_h(t, a)
    elif OP == OP_CELL_BWD_H:
        op_cell_bwd_h(t, a)
    elif OP == OP_CELL_FWD_SCAN:
        op_cell_fwd_scan(t, a)
    elif OP == OP_CELL_BWD_SCAN:
        op_cell_bwd_scan(t, a)
    elif OP == OP_GEMM_EPI:
        op_gemm_epi(t, a)
    elif OP == OP_COLSUM_DIV:
        op_colsum_div(t, a)
    elif OP == OP_GEMM_EPI_TAIL:
        op_gemm_epi_tail(t, a)
    elif OP == OP_GATHER_SEQ:
        op_gather_seq(t, a)
    elif OP == OP_GATHER_ROWS:
        op_gather_rows(t, a)
    elif OP == OP_MSE:
        op_mse(t, a)
    elif OP == OP_CE:
        op_ce(t, a)
    elif OP == OP_SUM:
        op_sum(t, a)
    elif OP == OP_OPT:
        op_opt(t, a)
    elif OP == OP_FILL:
        op_fill(t, a)
    elif OP == OP_COPY:
        op_copy(t, a)
    elif OP == OP_SEQ_OUT:
        op_seq_out(t, a)
    elif OP == OP_SOFTMAX:
        op_softmax(t, a)
    elif OP == OP_STL:
        op_stl(t, a)
    elif OP == OP_VAR_DESIGN:
        op_var_design(t, a)
    elif OP == OP_COLSCALE:
        op_colscale(t, a)
    elif OP == OP_CHOLSOLVE:
        op_cholsolve(t, a)
    elif OP == OP_ROWSCALE:
        op_rowscale(t, a)
    elif OP == OP_VAR_FORECAST:
        op_var_forecast(t, a)
    elif OP == OP_SUB:
        op_sub(t, a)
    elif OP == OP_SCALE:
        op_scale(t, a)
    elif OP == OP_ACT:
        op_act(t, a)
    elif OP == OP_ACT_BWD:
        op_act_bwd(t, a)
    elif OP == OP_MLP_ROWLOSS:
        op_mlp_rowloss(t, a)
    elif OP == OP_SUMSQ:
        op_sumsq(t, a)
    elif OP == OP_MLP_BLOSS:
        op_mlp_bloss(t, a)
    elif OP == OP_MLP_L2PART:
        op_mlp_l2part(t, a)
    elif OP == OP_MLP_L2FOLD:
        op_mlp_l2fold(t, a)
    elif OP == OP_MLP_ROWPART:
        op_mlp_rowpart(t, a)
    elif OP == OP_MLP_PERM:
        op_mlp_perm(t, a)
    elif OP == OP_MLP_EPOCH_LOSS:
        op_mlp_epoch_loss(t, a)
    elif OP == OP_MLP_LB_PART:
        op_mlp_lb_part(t, a)
    elif OP == OP_MLP_LB_FOLD:
        op_mlp_lb_fold(t, a)
    elif OP == OP_MLP_LB_EPOCH_PART:
        op_mlp_lb_epoch_part(t, a)
    elif OP == OP_MLP_LB_EPOCH_FOLD:
        op_mlp_lb_epoch_fold(t, a)
    elif OP == OP_SEQ_IOTA:
        op_seq_iota(t, a)
    elif OP == OP_PROPHET_PREP:
        op_prophet_prep(t, a)
    elif OP == OP_L2GRAD:
        op_l2grad(t, a)
    elif OP == OP_DIVS:
        op_divs(t, a)
    elif OP == OP_AF_ALPHA:
        op_af_alpha(t, a)
    elif OP == OP_AF_ROW:
        op_af_row(t, a)
    elif OP == OP_AF_COL:
        op_af_col(t, a)
    elif OP == OP_AF_RMEAN:
        op_af_rmean(t, a)
    elif OP == OP_AF_UPDATE_MAT:
        op_af_update_mat(t, a)
    elif OP == OP_AF_VEC:
        op_af_vec(t, a)
    elif OP == OP_AF_DENOM:
        op_af_denom(t, a)
    elif OP == OP_AF_APPLY:
        op_af_apply(t, a)
    elif OP == OP_SEG_SUMSQ:
        op_seg_sumsq(t, a)
    elif OP == OP_CHUNK_SUMSQ:
        op_chunk_sumsq(t, a)
    elif OP == OP_AF_BLK_SUMSQ:
        op_af_blk_sumsq(t, a)
    elif OP == OP_LAMB_UPD:
        op_lamb_upd(t, a)
    elif OP == OP_LAMB_RATIO:
        op_lamb_ratio(t, a)
    elif OP == OP_LAMB_APPLY:
        op_lamb_apply(t, a)
    elif OP == OP_LAMB_BLK:
        op_lamb_blk(t, a)
    elif OP == OP_LAMB_SEGFOLD:
        op_lamb_segfold(t, a)
    elif OP == OP_LAMB_CLIP:
        op_lamb_clip(t, a)
    elif OP == OP_LAMB_TRUST:
        op_lamb_trust(t, a)
    elif OP == OP_LAMB_APPLY_ALL:
        op_lamb_apply_all(t, a)
    elif OP == OP_LN_FWD:
        op_ln_fwd(t, a)
    elif OP == OP_LN_BWD_X:
        op_ln_bwd_x(t, a)
    elif OP == OP_LN_BWD_W:
        op_ln_bwd_w(t, a)
    elif OP == OP_THETA:
        op_theta(t, a)
    elif OP == OP_CROSTON:
        op_croston(t, a)
    elif OP == OP_ETS:
        op_ets(t, a)
    elif OP == OP_ETS_LIK:
        op_ets_lik(t, a)
    elif OP == OP_ETS_INIT:
        op_ets_init(t, a)
    elif OP == OP_GARCH:
        op_garch(t, a)
    elif OP == OP_PROPHET_FEATURES:
        op_prophet_features(t, a)
    elif OP == OP_PROPHET_FIT:
        op_prophet_fit(t, a)
    elif OP == OP_PROPHET_FG_PART:
        op_prophet_fg_part(t, a)
    elif OP == OP_PROPHET_FG_SUM:
        op_prophet_fg_sum(t, a)
    elif OP == OP_PROPHET_PREDICT:
        op_prophet_predict(t, a)
    elif OP == OP_MOE_ROUTE:
        op_moe_route(t, a)
    elif OP == OP_MOE_HIDDEN:
        op_moe_hidden(t, a)
    elif OP == OP_MOE_OUT:
        op_moe_out(t, a)
    # lane/apple-fast-tsa2: reached only under TSA2_VAR / TSA2_STL
    elif OP == OP_VAR_RESID:
        op_var_resid(t, a)
    elif OP == OP_VAR_SIGMA:
        op_var_sigma(t, a)
    elif OP == OP_STL_SEAS:
        op_stl_seas(t, a)
    elif OP == OP_STL_MA:
        op_stl_ma(t, a)
    elif OP == OP_STL_LOESS:
        op_stl_loess(t, a)
    elif OP == OP_STL_DESEAS:
        op_stl_deseas(t, a)
    elif OP == OP_STL_FINISH:
        op_stl_finish(t, a)
    elif OP == OP_ONE_HOT:
        op_one_hot(t, a)
    elif OP == OP_PROBA2:
        op_proba2(t, a)
