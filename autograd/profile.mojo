# SPDX-License-Identifier: Apache-2.0
# The tape's launch profile as text (lane ag-pass2-kernels, 2026-10-19).
# The counting lives in the executors (sequence/launch_profile.mojo); the
# tape marks which node kind issued the launches. Set MOJOLEARN_AG_PROFILE=1
# (counts) or 2 (counts and host-clock times between syncs: a relative
# profile, the syncs change the timing) before the executor and the tape
# are built. Every loop here walks profile rows, never data.
from sequence.launch_profile import (
    LaunchProfile, launch_profile_snapshot, launch_profile_reset, launch_profile_level,
    PROFILE_FORWARD, PROFILE_BACKWARD, PROFILE_OTHER, PROFILE_KIND_ALL,
)
from autograd.ops import (
    OP_AG_BIN_FWD, OP_AG_BIN_VJP, OP_AG_UNARY_FWD, OP_AG_UNARY_VJP, OP_AG_TRANSPOSE,
    OP_AG_REDUCE_PART, OP_AG_REDUCE_FOLD, OP_AG_REDUCE_VJP, OP_AG_SOFTMAX_FWD,
    OP_AG_SOFTMAX_VJP, OP_AG_ACCUMULATE, OP_AGK_BIN_VJP_PART, OP_AGK_GEMM_EPI, OP_AGF_LEVEL,
    LEAF, ADD, SUBTRACT, MULTIPLY, DIVIDE, RELU, SIGMOID, TANH, EXP, LOG, SQRT, SCALE,
    TRANSPOSE, RESHAPE, REDUCE, SOFTMAX, LOG_SOFTMAX, LOSS_ELEM, LOSS_CE, MATMUL, LINEAR, CUSTOM,
)
from autograd.data_ops import OP_AG_DROPOUT_FWD, OP_AG_DROPOUT_VJP, DROPOUT
from autograd.loss_ops import OP_LS_FIRST, OP_LS_LAST, OP_LS_ELEM_FWD, OP_LS_ELEM_VJP, OP_LS_CE_FWD, OP_LS_CE_VJP
from autograd.optim import OP_AGO_FIRST, OP_AGO_LAST
from autograd.model_ops import OP_AGM_FIRST, OP_AGM_LAST


def ag_op_name(op: Int) -> String:
    """A short name for an operation code in the profile."""
    if op == OP_AG_BIN_FWD:
        return "ag_bin_fwd"
    if op == OP_AG_BIN_VJP:
        return "ag_bin_vjp"
    if op == OP_AG_UNARY_FWD:
        return "ag_unary_fwd"
    if op == OP_AG_UNARY_VJP:
        return "ag_unary_vjp"
    if op == OP_AG_TRANSPOSE:
        return "ag_transpose"
    if op == OP_AG_REDUCE_PART:
        return "ag_reduce_part"
    if op == OP_AG_REDUCE_FOLD:
        return "ag_reduce_fold"
    if op == OP_AG_REDUCE_VJP:
        return "ag_reduce_vjp"
    if op == OP_AG_SOFTMAX_FWD:
        return "ag_softmax_fwd"
    if op == OP_AG_SOFTMAX_VJP:
        return "ag_softmax_vjp"
    if op == OP_AG_ACCUMULATE:
        return "ag_accumulate"
    if op == OP_AGK_BIN_VJP_PART:
        return "ag_bin_vjp_part"
    if op == OP_AGK_GEMM_EPI:
        return "ag_gemm_epi"
    if op == OP_AGF_LEVEL:
        return "ag_fold_level"
    if op == OP_AG_DROPOUT_FWD:
        return "ag_dropout_fwd"
    if op == OP_AG_DROPOUT_VJP:
        return "ag_dropout_vjp"
    if op == OP_LS_ELEM_FWD:
        return "loss_elem_fwd"
    if op == OP_LS_ELEM_VJP:
        return "loss_elem_vjp"
    if op == OP_LS_CE_FWD:
        return "loss_ce_fwd"
    if op == OP_LS_CE_VJP:
        return "loss_ce_vjp"
    if op >= OP_LS_FIRST and op <= OP_LS_LAST:
        return "loss_metric_" + String(op)
    if op >= OP_AGO_FIRST and op <= OP_AGO_LAST:
        return "optim_" + String(op)
    if op >= OP_AGM_FIRST and op <= OP_AGM_LAST:
        return "model_" + String(op)
    # sequence/ops.mojo codes the tape and the trainers launch
    if op == 1:
        return "gemm"
    if op == 3:
        return "colsum"
    if op == 11:
        return "opt_step"
    if op == 12:
        return "fill"
    if op == 13:
        return "copy"
    if op == 15:
        return "softmax"
    if op == 24:
        return "act"
    if op == 25:
        return "act_bwd"
    if op == 29:
        return "l2grad"
    if op == 30:
        return "divs"
    if op == 60:
        return "gemm_epi"
    if op == 61:
        return "colsum_div"
    if op == 62:
        return "gemm_epi_tail"
    # the MLP loss block (sequence/ops.mojo OP_MLP_LB_*, 1600..1603)
    if op == 1600:
        return "mlp_lb_part"
    if op == 1601:
        return "mlp_lb_fold"
    if op == 1602:
        return "mlp_lb_epoch_part"
    if op == 1603:
        return "mlp_lb_epoch_fold"
    return "op_" + String(op)


def ag_kind_name(kind: Int) -> String:
    """A short name for a tape node kind in the profile."""
    if kind == PROFILE_KIND_ALL:
        return "all"
    if kind == LEAF:
        return "leaf"
    if kind == ADD:
        return "add"
    if kind == SUBTRACT:
        return "subtract"
    if kind == MULTIPLY:
        return "multiply"
    if kind == DIVIDE:
        return "divide"
    if kind == RELU:
        return "relu"
    if kind == SIGMOID:
        return "sigmoid"
    if kind == TANH:
        return "tanh"
    if kind == EXP:
        return "exp"
    if kind == LOG:
        return "log"
    if kind == SQRT:
        return "sqrt"
    if kind == SCALE:
        return "scale"
    if kind == TRANSPOSE:
        return "transpose"
    if kind == RESHAPE:
        return "reshape"
    if kind == REDUCE:
        return "reduce"
    if kind == SOFTMAX:
        return "softmax"
    if kind == LOG_SOFTMAX:
        return "log_softmax"
    if kind == LOSS_ELEM:
        return "loss_elem"
    if kind == LOSS_CE:
        return "loss_ce"
    if kind == MATMUL:
        return "matmul"
    if kind == LINEAR:
        return "linear"
    if kind == DROPOUT:
        return "dropout"
    if kind >= CUSTOM:
        return "custom_" + String(kind - CUSTOM)
    return "kind_" + String(kind)


def _phase_name(phase: Int) -> String:
    if phase == PROFILE_FORWARD:
        return "forward"
    if phase == PROFILE_BACKWARD:
        return "backward"
    return "other"


def _ms(ns: Int) -> String:
    """ns as milliseconds with three decimals."""
    var us = ns // 1000
    var whole = us // 1000
    var frac = us - whole * 1000
    var f = String(frac)
    if frac < 10:
        f = "00" + f
    elif frac < 100:
        f = "0" + f
    return String(whole) + "." + f


def ag_profile_report() raises -> String:
    """The process-wide launch profile as text: one OP line per operation
    code (launches, elements, and at level 2 milliseconds), one KIND line
    per (phase, node kind) the tape marked, and the launches not yet marked."""
    var p = launch_profile_snapshot()
    var level = max(p.level, launch_profile_level())
    var total = 0
    var total_ns = 0
    for i in range(len(p.ops)):  # driver-loop: over profiled operation codes
        total += p.op_launches[i]
        total_ns += p.op_ns[i]
    var out = String("AG_PROFILE level=") + String(level) + " launches=" + String(total)
    if level >= 2:
        out += " ms=" + _ms(total_ns) + " (host clock between syncs: a relative profile; the syncs change the timing)"
    out += "\n"
    for i in range(len(p.ops)):  # driver-loop: over profiled operation codes
        out += "AG_PROFILE OP " + ag_op_name(p.ops[i]) + " code=" + String(p.ops[i])
        out += " launches=" + String(p.op_launches[i]) + " elements=" + String(p.op_elements[i])
        if level >= 2:
            out += " ms=" + _ms(p.op_ns[i])
        out += "\n"
    for i in range(len(p.row_kind)):  # driver-loop: over profile rows
        out += "AG_PROFILE KIND " + _phase_name(p.row_phase[i]) + " " + ag_kind_name(p.row_kind[i])
        out += " launches=" + String(p.row_launches[i])
        if level >= 2:
            out += " ms=" + _ms(p.row_ns[i])
        out += "\n"
    if p.pending_launches > 0:
        out += "AG_PROFILE KIND unmarked all launches=" + String(p.pending_launches)
        if level >= 2:
            out += " ms=" + _ms(p.pending_ns)
        out += "\n"
    return out


def ag_profile_on() -> Bool:
    """True when MOJOLEARN_AG_PROFILE asks for a profile."""
    return launch_profile_level() > 0


def ag_profile_reset() raises:
    """Clear the process-wide launch profile."""
    launch_profile_reset()
