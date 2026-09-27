# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`apply[OP]`: the one table from an operation code to its element body, read
by both executors (`sequence/exec.mojo`, `sequence/exec_device.mojo`)."""
from sequence.ops import (
    Args,
    OP_GEMM,
    OP_BIAS,
    OP_COLSUM,
    OP_CELL_FWD,
    OP_CELL_BWD,
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
    OP_LAMB_UPD,
    OP_LAMB_RATIO,
    OP_LAMB_APPLY,
    OP_LN_FWD,
    OP_LN_BWD_X,
    OP_LN_BWD_W,
    op_gemm,
    op_bias,
    op_colsum,
    op_cell_fwd,
    op_cell_bwd,
    op_gather_seq,
    op_gather_rows,
    op_mse,
    op_ce,
    op_sum,
    op_opt,
    op_fill,
    op_copy,
    op_seq_out,
    op_softmax,
)
from sequence.adafactor import op_af_alpha, op_af_row, op_af_col, op_af_rmean, op_af_update_mat, op_af_vec, op_af_denom, op_af_apply, op_seg_sumsq, op_lamb_upd, op_lamb_ratio, op_lamb_apply
from sequence.layernorm import op_ln_bwd_w, op_ln_bwd_x, op_ln_fwd
from sequence.mlp import op_act, op_act_bwd, op_divs, op_l2grad, op_mlp_bloss, op_mlp_rowloss, op_sumsq
from sequence.stl import op_stl
from sequence.vecar import op_cholsolve, op_colscale, op_rowscale, op_scale, op_sub, op_var_design, op_var_forecast


@always_inline
def apply[OP: Int](t: Int, a: Args):
    comptime if OP == OP_GEMM:
        op_gemm(t, a)
    elif OP == OP_BIAS:
        op_bias(t, a)
    elif OP == OP_COLSUM:
        op_colsum(t, a)
    elif OP == OP_CELL_FWD:
        op_cell_fwd(t, a)
    elif OP == OP_CELL_BWD:
        op_cell_bwd(t, a)
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
    elif OP == OP_LAMB_UPD:
        op_lamb_upd(t, a)
    elif OP == OP_LAMB_RATIO:
        op_lamb_ratio(t, a)
    elif OP == OP_LAMB_APPLY:
        op_lamb_apply(t, a)
    elif OP == OP_LN_FWD:
        op_ln_fwd(t, a)
    elif OP == OP_LN_BWD_X:
        op_ln_bwd_x(t, a)
    elif OP == OP_LN_BWD_W:
        op_ln_bwd_w(t, a)
