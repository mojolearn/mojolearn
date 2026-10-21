# SPDX-License-Identifier: Apache-2.0
"""Typed code moved word for word from bindings/_mojolearn_training_host.mojo."""
from training.neural_arithmetic_profile import neural_training_profile
from training.neural_session_mlp_host import nn_mlp_sessions_host
from std.os import abort
from bindings.hostptr import f32_ptr, i32_ptr, read_f32, read_i32
from checks.kernel_matrix import (
    COLUMN_CPU,
    TARGET_COLUMN,
    column_name,
)
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from training.checks.loss_oracle import (
    REDUCTION_MEAN,
    REDUCTION_NONE,
    REDUCTION_SUM,
    CeConfig,
    ce_backward_oracle,
    ce_forward_oracle,
)
from training.maximize import maximize_negate
from training.checks.optimizer_oracle import (
    OPT_ADAM,
    OPT_ADAMW,
    OPT_SGD,
    OptimizerConfig,
    clip_grad_norm_oracle,
    microbatch_split_is_identical,
    optimizer_step_oracle,
    refuse_nonfinite_scalar,
)
from training.host.samba_ops_oracle import (
    host_samba_accumulate,
    host_samba_embedding_backward,
    host_samba_embedding_forward,
    host_samba_linear_backward,
    host_samba_linear_forward,
    host_samba_rms_norm_backward,
    host_samba_rms_norm_forward,
)
from training.checks.chunked_lm_head_oracle import (
    chunked_lm_head_v2_oracle,
    chunked_lm_head_v2_oracle_forward,
)
from mamba.host.device_shim import DeviceContext as HostDeviceContext
from mamba.host.gen.philox_neural import neural_rng_host
from training.host.mlp_oracle import (
    MLP_ORACLE_HOST_SABOTAGE,
    host_mlp_bias_activation,
    host_mlp_relu_backward,
    host_mlp_sum_rows,
    host_mlp_validate_shape,
)
from training.neural_identical_experiments import IDN_CHUNKED_LM_HEAD_V2
from gemm.contract import CONTRACT_K_LEAF_MIN


def _host_offsets(offsets_ptr: MutPointer[Int32, MutUntrackedOrigin], n_tensors: Int) raises -> List[Int]:
    """`_offsets_from_ptr`, `training/estimator.mojo:131`, in its words."""
    if n_tensors < 1:
        raise Error(
            String("mojolearn training: n_tensors must be at least 1, got ")
            + String(n_tensors)
        )
    var offsets = List[Int]()
    for j in range(n_tensors + 1):
        offsets.append(Int(offsets_ptr[j]))
    if offsets[0] != 0:
        raise Error(
            String("mojolearn training: offsets[0] is ")
            + String(offsets[0])
            + String(", must be 0 (offsets index the flat buffer from its")
            + String(" first element; optimizer contract 3.3)")
        )
    for j in range(n_tensors):
        var count = offsets[j + 1] - offsets[j]
        if count < 1:
            raise Error(
                String("mojolearn training: tensor ")
                + String(j)
                + String(" spans ")
                + String(count)
                + String(" elements (offsets ")
                + String(offsets[j])
                + String(" .. ")
                + String(offsets[j + 1])
                + String("); offsets must be strictly ascending and an empty")
                + String(" tensor is REFUSED, not skipped")
            )
    return offsets^


def _host_refuse_hyperparameters(cfg: OptimizerConfig) raises:
    """`_refuse_hyperparameters`, `training/estimator.mojo:186`."""
    refuse_nonfinite_scalar(String("lr"), cfg.lr)
    refuse_nonfinite_scalar(String("beta1"), cfg.beta1)
    refuse_nonfinite_scalar(String("beta2"), cfg.beta2)
    refuse_nonfinite_scalar(String("eps"), cfg.eps)
    refuse_nonfinite_scalar(String("weight_decay"), cfg.weight_decay)
    refuse_nonfinite_scalar(String("momentum"), cfg.momentum)
    refuse_nonfinite_scalar(String("dampening"), cfg.dampening)
    refuse_nonfinite_scalar(String("max_norm"), cfg.max_norm)
    if cfg.kind != OPT_SGD:
        if cfg.beta1 < Float32(0.0) or cfg.beta1 >= Float32(1.0):
            raise Error(
                String("mojolearn training: beta1 must be in [0, 1), got ")
                + String(cfg.beta1)
                + String(" (at beta1 = 1 the bias correction 1 - beta1^t is")
                + String(" exactly 0 and step_scalars divides by it)")
            )
        if cfg.beta2 < Float32(0.0) or cfg.beta2 >= Float32(1.0):
            raise Error(
                String("mojolearn training: beta2 must be in [0, 1), got ")
                + String(cfg.beta2)
            )
        if cfg.eps < Float32(0.0):
            raise Error(
                String("mojolearn training: eps must be >= 0, got ")
                + String(cfg.eps)
            )


def _write_f32(values: List[Float32], dst: MutPointer[Float32, MutUntrackedOrigin]):
    for i in range(len(values)):
        dst[i] = values[i]


def neural_rng_host_into(
    a0: Int, a1: Int, n: Int, offset: Int, seed_lo: Int, seed_hi: Int, stream_id: Int, kind: Int,
    pa: Float32, pb: Float32,
) raises -> Int:
    var count = 0
    var ctx = HostDeviceContext()
    count = neural_rng_host(
        ctx, f32_ptr(a0), f32_ptr(a1), n, offset, seed_lo, seed_hi,
        stream_id, kind, pa, pb,
    )
    return count


def clip_grad_norm_host_into(gp: MutPointer[Float32, MutUntrackedOrigin], op: MutPointer[Int32, MutUntrackedOrigin], fp: MutPointer[Float32, MutUntrackedOrigin], n_tensors: Int, max_norm: Float32) raises -> Int:
    var n_total = 0
    if max_norm <= Float32(0.0):
        raise Error(
            String("mojolearn training: max_norm must be > 0, got ")
            + String(max_norm)
            + String("; this entry point IS the clip, and 'no clipping' is")
            + String(" spelled by not calling it (the optimizer step takes")
            + String(" max_norm <= 0 as OFF because it has another job)")
        )
    refuse_nonfinite_scalar(String("max_norm"), max_norm)
    var offsets = _host_offsets(op, n_tensors)
    n_total = offsets[n_tensors]
    var grads = List[Float32](length=n_total, fill=Float32(0.0))
    for i in range(n_total):
        grads[i] = gp[i]
    var sumsq = List[Float32]()
    var norms = List[Float32]()
    var total = List[Float32]()
    _ = clip_grad_norm_oracle(grads, offsets, max_norm, sumsq, norms, total)
    for i in range(n_total):
        gp[i] = grads[i]
    fp[0] = total[1]
    fp[1] = total[2]
    return n_total


def mlp_sum_rows_host_into(x_address: Int, op: MutPointer[Float32, MutUntrackedOrigin], rows: Int, cols: Int) raises -> Int:
    var count = 0
    var x = read_f32(x_address, rows * cols)
    var out = host_mlp_sum_rows(x, rows, cols)
    for i in range(len(out)):
        op[i] = out[i]
    count = len(out)
    return count


def mlp_relu_backward_host_into(a_address: Int, g_address: Int, op: MutPointer[Float32, MutUntrackedOrigin], rows: Int, cols: Int) raises -> Int:
    var count = 0
    var a = read_f32(a_address, rows * cols)
    var g = read_f32(g_address, rows * cols)
    var out = host_mlp_relu_backward(a, g, rows, cols)
    for i in range(len(out)):
        op[i] = out[i]
    count = len(out)
    return count


def mlp_bias_activation_host_into(x_address: Int, b_address: Int, op: MutPointer[Float32, MutUntrackedOrigin], rows: Int, cols: Int, relu_flag: Int) raises -> Int:
    var count = 0
    var x = read_f32(x_address, rows * cols)
    var b = read_f32(b_address, cols)
    var out = host_mlp_bias_activation(x, b, rows, cols, relu_flag)
    for i in range(len(out)):
        op[i] = out[i]
    count = len(out)
    return count


def ce_loss_host_into(
    lp: MutPointer[Float32, MutUntrackedOrigin], rp: MutPointer[Float32, MutUntrackedOrigin], dp: MutPointer[Float32, MutUntrackedOrigin], x_address: Int, t_address: Int, n_rows: Int, vocab: Int,
    ignore_index: Int, reduction: Int, num_items: Int, want_grad: Int, label_smoothing: Float32,
) raises -> Int:
    var count = 0
    if reduction != REDUCTION_NONE:
        if reduction != REDUCTION_SUM and reduction != REDUCTION_MEAN:
            raise Error(
                String("mojolearn training: reduction must be 0 (none), 1")
                + String(" (sum) or 2 (mean), got ")
                + String(reduction)
            )
    if want_grad != 0 and reduction == REDUCTION_NONE:
        raise Error(
            String("mojolearn training: REDUCTION_NONE has no backward")
            + String(" (loss contract section 11); ask for a gradient with")
            + String(" reduction 'sum' or 'mean'")
        )
    if n_rows < 1:
        raise Error(
            String("mojolearn training: n_rows must be at least 1, got ")
            + String(n_rows)
        )
    if vocab < 1:
        raise Error(
            String("mojolearn training: vocab must be at least 1, got ")
            + String(vocab)
        )
    var cfg = CeConfig(vocab, ignore_index, reduction, label_smoothing, num_items)
    var logits = read_f32(x_address, n_rows * vocab)
    var targets = read_i32(t_address, n_rows)
    var st = ce_forward_oracle(logits, targets, cfg)
    count = st.count
    if want_grad != 0:
        ce_backward_oracle(st, targets, cfg)
    for i in range(n_rows):
        rp[i] = st.row[i]
    if reduction != REDUCTION_NONE:
        lp[0] = st.loss[0]
    else:
        lp[0] = Float32(0.0)
    if want_grad != 0:
        for i in range(n_rows * vocab):
            dp[i] = st.dlogits[i]
    return count


def optimizer_step_host_into(
    pp: MutPointer[Float32, MutUntrackedOrigin], gp: MutPointer[Float32, MutUntrackedOrigin], mp: MutPointer[Float32, MutUntrackedOrigin], vp: MutPointer[Float32, MutUntrackedOrigin], op: MutPointer[Int32, MutUntrackedOrigin], ip: MutPointer[Int32, MutUntrackedOrigin], fp: MutPointer[Float32, MutUntrackedOrigin],
    n_tensors: Int, kind: Int, t: Int, nesterov: Int, lr: Float32, beta1: Float32, beta2: Float32,
    eps: Float32, weight_decay: Float32, momentum: Float32, dampening: Float32, max_norm: Float32,
    maximize: Bool,
) raises -> Int:
    var n_total = 0
    if kind != OPT_SGD and kind != OPT_ADAM and kind != OPT_ADAMW:
        raise Error(
            String("mojolearn training: kind must be 0 (SGD), 1 (Adam) or 2")
            + String(" (AdamW), got ")
            + String(kind)
        )
    if t < 1:
        raise Error(
            String("mojolearn training: t is ONE-BASED and the first step of")
            + String(" a run is t = 1, got ")
            + String(t)
            + String(" (at t = 0 the bias correction 1 - beta^0 is exactly 0")
            + String(" and step_scalars divides by it)")
        )
    var offsets = _host_offsets(op, n_tensors)
    n_total = offsets[n_tensors]
    var cfg = OptimizerConfig(
        kind, lr, beta1, beta2, eps, weight_decay, momentum, dampening,
        nesterov != 0, max_norm,
    )
    _host_refuse_hyperparameters(cfg)
    var param = List[Float32](length=n_total, fill=Float32(0.0))
    var grad = List[Float32](length=n_total, fill=Float32(0.0))
    var m_state = List[Float32](length=n_total, fill=Float32(0.0))
    var v_state = List[Float32](length=n_total, fill=Float32(0.0))
    for i in range(n_total):
        param[i] = pp[i]
        grad[i] = maximize_negate(gp[i]) if maximize else gp[i]
        m_state[i] = mp[i]
        v_state[i] = vp[i]
    var buf_initialized = List[Bool]()
    for j in range(n_tensors):
        buf_initialized.append(ip[j] != Int32(0))
    var stages = optimizer_step_oracle(
        param, grad, m_state, v_state, buf_initialized, offsets, cfg, t
    )
    for i in range(n_total):
        pp[i] = param[i]
        mp[i] = m_state[i]
        vp[i] = v_state[i]
    if max_norm > Float32(0.0):
        for i in range(n_total):
            gp[i] = maximize_negate(grad[i]) if maximize else grad[i]
    for j in range(n_tensors):
        ip[j] = Int32(1) if buf_initialized[j] else Int32(0)
    fp[0] = Float32(0.0)
    fp[1] = Float32(0.0)
    fp[2] = Float32(0.0)
    if max_norm > Float32(0.0):
        fp[0] = Float32(1.0)
        fp[1] = stages.clip_total[1]
        fp[2] = stages.clip_total[2]
    return n_total
