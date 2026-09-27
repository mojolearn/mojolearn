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
from sequence.stl import op_stl


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
