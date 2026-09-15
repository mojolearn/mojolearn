# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding for the `_mojolearn_training` family: the small MLP's training
step (the mlp lane, 2026-09-14; brief docs/lanes/BRIEF_cpu_training_2026-09-13.md
sections 1.1 and 3.2).

HOST ONLY. No DeviceContext, no kernel launch, no GPU. `SmallMLPTrainer`
(`python/mojolearn/_mlp_impl.py`) composes a training step from the pinned
GEMM (`_mojolearn_linalg`, already served by the linalg host binding), the
three small MLP operations, the cross-entropy loss and its gradient, and one
AdamW update. This binding carries the last three under the GPU training
binding's names (`bindings/_mojolearn_training.mojo`), with the SAME address
contracts and params lists, so the trainer runs unchanged on a CPU-only
install through `_backend._HOST_MODULES` (`"_mojolearn_training":
"_mojolearn_training_host"`), the way `LanguageModelHostTrainer` composes the
byte LM step from the same normative oracles:

  `mlp_bias_activation`, `mlp_relu_backward`, `mlp_sum_rows`
                  `training/host/mlp_oracle.mojo`, `training/mlp_ops.mojo`
                  restated.
  `ce_loss`       `training/checks/loss_oracle.mojo::ce_forward_oracle` and
                  `ce_backward_oracle`, THE NORMATIVE ANSWER of
                  `mojolearn.identical.loss.ce.fp32.v1` the device loss is
                  gated against, in `identical_ce_loss_host`'s order and with
                  its refusals (`training/estimator.mojo:576`).
  `optimizer_step`
                  `training/checks/optimizer_oracle.mojo::optimizer_step_oracle`,
                  THE NORMATIVE ANSWER of `mojolearn.identical.optimizer.fp32.v1`,
                  behind `identical_optimizer_step_host`'s refusals
                  (`training/estimator.mojo:235`: the kind, the one-based `t`,
                  the offsets registry, the hyperparameters) and its `info`
                  convention.

Every other entry of the GPU training binding (the clip on its own, the
accumulation, the Samba stack's operations, the neural RNG and the
multi-GPU availability probes) is deliberately ABSENT, so those surfaces
refuse BY NAME through `_HostBinding`.

The sabotage arm (`training_host_sabotage`) is
`training/host/mlp_oracle.mojo::MLP_ORACLE_HOST_SABOTAGE`: every row sum is
walked descending.
"""
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

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
from training.checks.optimizer_oracle import (
    OPT_ADAM,
    OPT_ADAMW,
    OPT_SGD,
    OptimizerConfig,
    optimizer_step_oracle,
    refuse_nonfinite_scalar,
)
from training.host.mlp_oracle import (
    MLP_ORACLE_HOST_SABOTAGE,
    host_mlp_bias_activation,
    host_mlp_relu_backward,
    host_mlp_sum_rows,
    host_mlp_validate_shape,
)


def _index(value: PythonObject) raises -> Int:
    var type_name = String(py=value.__class__.__name__)
    if type_name == "bool" or type_name == "bool_":
        raise Error("training host: integers expected, not booleans")
    var operator_module = Python.import_module("operator")
    return Int(py=operator_module.index(value))


def training_host_numeric_mode_binding() raises -> PythonObject:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL, (
        "training host: this binding restates the IDENTICAL arms only;"
        " bindings/build_host_family.sh passes -D MOJOLEARN_NUMERIC_IDENTICAL=1"
    )
    return PythonObject(GLOBAL_NUMERIC_MODE)


def training_host_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def training_host_column_binding() raises -> PythonObject:
    """`column_name(TARGET_COLUMN)`, the comptime assert's witness: "cpu".
    THE COLUMN IS THE CPU COLUMN, OR THIS DOES NOT BUILD; the assert lives
    in a function body PyInit registers, so it is compiled in every build.
    `bindings/build_host_family.sh` passes -D MOJOLEARN_COLUMN_CPU."""
    comptime assert TARGET_COLUMN == COLUMN_CPU, (
        "training host: this binding compiles the CPU column only; pass"
        " -D MOJOLEARN_COLUMN_CPU (bindings/build_training_host.sh does)"
    )
    return PythonObject(column_name(TARGET_COLUMN))


def training_host_sabotage_binding() raises -> PythonObject:
    """Whether this binary walks every MLP row sum descending on purpose
    (-D MOJOLEARN_HOST_SABOTAGE=1, the gate's negative control)."""
    return PythonObject(MLP_ORACLE_HOST_SABOTAGE)


def training_numeric_mode_binding() raises -> PythonObject:
    """THE BUILD'S TIER as the `NUMERIC_*` code: always 1 here."""
    return PythonObject(GLOBAL_NUMERIC_MODE)


def training_vendor_binding() raises -> PythonObject:
    """"cpu", the answer `_backend.read_vendor` expects from a host binding."""
    return PythonObject(String("cpu"))


# ===========================================================================
# THE OPTIMIZER STEP
# ===========================================================================


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


def optimizer_step_binding(
    param_addr: PythonObject,
    grad_addr: PythonObject,
    m_addr: PythonObject,
    v_addr: PythonObject,
    offsets_addr: PythonObject,
    init_addr: PythonObject,
    info_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """One step of `mojolearn.identical.optimizer.fp32.v1` on the host.
    Returns `N = offsets[J]`. `params`: `0 n_tensors, 1 kind, 2 t, 3 nesterov,
    4 lr, 5 beta1, 6 beta2, 7 eps, 8 weight_decay, 9 momentum, 10 dampening,
    11 max_norm` (the GPU binding's list, word for word)."""
    if len(params) != 12:
        raise Error(
            "optimizer_step: params must contain 12 values, got "
            + String(len(params))
        )
    var pp = f32_ptr(_index(param_addr))
    var gp = f32_ptr(_index(grad_addr))
    var mp = f32_ptr(_index(m_addr))
    var vp = f32_ptr(_index(v_addr))
    var op = i32_ptr(_index(offsets_addr))
    var ip = i32_ptr(_index(init_addr))
    var fp = f32_ptr(_index(info_addr))
    var n_tensors = _index(params[0])
    var kind = _index(params[1])
    var t = _index(params[2])
    var nesterov = _index(params[3])
    var lr = Float32(Float64(py=params[4]))
    var beta1 = Float32(Float64(py=params[5]))
    var beta2 = Float32(Float64(py=params[6]))
    var eps = Float32(Float64(py=params[7]))
    var weight_decay = Float32(Float64(py=params[8]))
    var momentum = Float32(Float64(py=params[9]))
    var dampening = Float32(Float64(py=params[10]))
    var max_norm = Float32(Float64(py=params[11]))
    var n_total = 0
    with GILReleased(Python()):
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
            grad[i] = gp[i]
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
                gp[i] = grad[i]
        for j in range(n_tensors):
            ip[j] = Int32(1) if buf_initialized[j] else Int32(0)
        fp[0] = Float32(0.0)
        fp[1] = Float32(0.0)
        fp[2] = Float32(0.0)
        if max_norm > Float32(0.0):
            fp[0] = Float32(1.0)
            fp[1] = stages.clip_total[1]
            fp[2] = stages.clip_total[2]
    return PythonObject(n_total)


# ===========================================================================
# THE CROSS-ENTROPY LOSS
# ===========================================================================


def ce_loss_binding(
    loss_addr: PythonObject,
    row_addr: PythonObject,
    dlogits_addr: PythonObject,
    logits_addr: PythonObject,
    targets_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`mojolearn.identical.loss.ce.fp32.v1` on the host, forward and
    optionally backward, in ONE call. Returns `count`. `params`: `0 n_rows,
    1 vocab, 2 ignore_index, 3 reduction, 4 num_items, 5 want_grad,
    6 label_smoothing` (the GPU binding's list, word for word)."""
    if len(params) != 7:
        raise Error(
            "ce_loss: params must contain 7 values, got " + String(len(params))
        )
    var lp = f32_ptr(_index(loss_addr))
    var rp = f32_ptr(_index(row_addr))
    var dp = f32_ptr(_index(dlogits_addr))
    var x_address = _index(logits_addr)
    var t_address = _index(targets_addr)
    var n_rows = _index(params[0])
    var vocab = _index(params[1])
    var ignore_index = _index(params[2])
    var reduction = _index(params[3])
    var num_items = _index(params[4])
    var want_grad = _index(params[5])
    var label_smoothing = Float32(Float64(py=params[6]))
    var count = 0
    with GILReleased(Python()):
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
    return PythonObject(count)


# ===========================================================================
# THE SMALL MLP'S THREE OPERATIONS
# ===========================================================================


def mlp_bias_activation_binding(
    input_addr: PythonObject, bias_addr: PythonObject,
    out_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """C-row-major f32 bias + optional ReLU; params=[rows,cols,relu_flag].
    Returns rows*cols; IDENTICAL only."""
    if len(params) != 3:
        raise Error("mlp_bias_activation params must be [rows,cols,relu_flag]")
    var rows = _index(params[0])
    var cols = _index(params[1])
    var relu_flag = _index(params[2])
    host_mlp_validate_shape(rows, cols)
    if relu_flag != 0 and relu_flag != 1:
        raise Error("mlp_bias_activation relu_flag must be 0 or 1")
    var x_address = _index(input_addr)
    var b_address = _index(bias_addr)
    var op = f32_ptr(_index(out_addr))
    var count = 0
    with GILReleased(Python()):
        var x = read_f32(x_address, rows * cols)
        var b = read_f32(b_address, cols)
        var out = host_mlp_bias_activation(x, b, rows, cols, relu_flag)
        for i in range(len(out)):
            op[i] = out[i]
        count = len(out)
    return PythonObject(count)


def mlp_relu_backward_binding(
    activation_addr: PythonObject, incoming_addr: PythonObject,
    out_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """params=[rows,cols]; the derivative is incoming where activation > 0,
    otherwise +0. Returns rows*cols; IDENTICAL only."""
    if len(params) != 2:
        raise Error("mlp_relu_backward params must be [rows,cols]")
    var rows = _index(params[0])
    var cols = _index(params[1])
    host_mlp_validate_shape(rows, cols)
    var a_address = _index(activation_addr)
    var g_address = _index(incoming_addr)
    var op = f32_ptr(_index(out_addr))
    var count = 0
    with GILReleased(Python()):
        var a = read_f32(a_address, rows * cols)
        var g = read_f32(g_address, rows * cols)
        var out = host_mlp_relu_backward(a, g, rows, cols)
        for i in range(len(out)):
            op[i] = out[i]
        count = len(out)
    return PythonObject(count)


def mlp_sum_rows_binding(
    input_addr: PythonObject, out_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """params=[rows,cols]; returns cols after the ascending row sum;
    IDENTICAL only."""
    if len(params) != 2:
        raise Error("mlp_sum_rows params must be [rows,cols]")
    var rows = _index(params[0])
    var cols = _index(params[1])
    host_mlp_validate_shape(rows, cols)
    var x_address = _index(input_addr)
    var op = f32_ptr(_index(out_addr))
    var count = 0
    with GILReleased(Python()):
        var x = read_f32(x_address, rows * cols)
        var out = host_mlp_sum_rows(x, rows, cols)
        for i in range(len(out)):
            op[i] = out[i]
        count = len(out)
    return PythonObject(count)


@export
def PyInit__mojolearn_training_host() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_training_host")
        module.def_function[training_host_numeric_mode_binding]("training_host_numeric_mode")
        module.def_function[training_host_vendor_binding]("training_host_vendor")
        module.def_function[training_host_column_binding]("training_host_column")
        module.def_function[training_host_sabotage_binding]("training_host_sabotage")
        module.def_function[training_numeric_mode_binding]("training_numeric_mode")
        module.def_function[training_vendor_binding]("training_vendor")
        module.def_function[optimizer_step_binding]("optimizer_step")
        module.def_function[ce_loss_binding]("ce_loss")
        module.def_function[mlp_bias_activation_binding]("mlp_bias_activation")
        module.def_function[mlp_relu_backward_binding]("mlp_relu_backward")
        module.def_function[mlp_sum_rows_binding]("mlp_sum_rows")
        return module.finalize()
    except error:
        abort(String("failed to create _mojolearn_training_host: ", error))
