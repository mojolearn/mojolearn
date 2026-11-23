# SPDX-License-Identifier: Apache-2.0
"""The activation layers of a Sequential description and the identity layer (lane nn-activations, 2026-11-23).

Each layer is a declaration over an EXISTING tape op of the import family (autograd/import_api.mojo) whose element
bodies are checks/numerics.mojo's activation bodies (one body each, called by the host column and every device
kernel); autograd/model_blocks.mojo dispatches these codes here between its `# nn-activations` markers:

  code  layer                          tape op                                  launches forward / backward
  47    identity                       none: the layer returns its input        0 / 0
  48    leaky_relu (f negative_slope)  imp_unary IMP_LEAKY_RELU, ImportRules 192 1 / 1
  49    elu (f alpha)                  imp_unary IMP_ELU                        1 / 1
  50    prelu (a0 num_parameters,      imp_binary IMP_PRELU against the slope   1 / 2 + the slope's fold
        a1 the init's float32 bits)    viewed (C, 1, ..) on the sample's first  (1 launch for at most 64
                                       axis (or (1,) shared), ImportRules 193   terms a slope, else 2)
  51    mish                           imp_unary IMP_MISH                       1 / 1
  52    hardswish                      imp_unary IMP_HARD_SWISH (1/6, 1/2)      1 / 1
  53    hardsigmoid                    imp_unary IMP_HARD_SIGMOID (1/6, 1/2)    1 / 1

An activation is one elementwise launch forward and one backward, as silu, gelu and softplus are (codes 32..34);
the model interpreter fuses none of them into the linear node before them (only relu, sigmoid, tanh and softmax
are, autograd/model.mojo _absorbed_activation), so neither is fused here. PReLU's slope gradient is the sum of its
elements' terms over the batch and the other axes: the broadcast fold of the binary kinds (OP_RN_REDUCE_PART's
64-term leaves, then the blocked fold), the same order on every executor. Per forward the binary kind writes its
small shape descriptor into tape storage, as every imp_binary record does. Nothing is looked up, allocated or
launched per element; the options are read from the description when the layer is recorded.
"""
from std.math import isfinite
from std.memory import bitcast
from sequence.exec_trait import Exec
from sequence.ops import FP
from autograd.tape import Tape, Tensor
from autograd.rankn import ShapeN
from autograd.rankn_api import rn_shape
from autograd.import_api import imp_unary, imp_binary
from autograd.import_ops import (
    IMP_LEAKY_RELU, IMP_ELU, IMP_MISH, IMP_HARD_SWISH, IMP_HARD_SIGMOID, IMP_PRELU,
)

comptime LAYER_IDENTITY = 47
comptime LAYER_LEAKY_RELU = 48
comptime LAYER_ELU = 49
comptime LAYER_PRELU = 50
comptime LAYER_MISH = 51
comptime LAYER_HARDSWISH = 52
comptime LAYER_HARDSIGMOID = 53
comptime NN_ACT_FIRST = 47
comptime NN_ACT_LAST = 53

#: PyTorch's Hardswish / Hardsigmoid ramp: clamp(x / 6 + 1 / 2, 0, 1), 1/6 rounded once to float32
comptime NN_ACT_SIXTH = Float32(0.16666666666666666)
comptime NN_ACT_HALF = Float32(0.5)


def is_nn_activation(code: Int) -> Bool:
    return code >= NN_ACT_FIRST and code <= NN_ACT_LAST


def prelu_init(a1: Int) -> Float32:
    """A PReLU layer's initial slope: its a1 holds the float32's bits."""
    return bitcast[DType.float32](Int32(a1))


def nn_act_shape(code: Int, a0: Int, a1: Int, f: Float32, cur: ShapeN, name: String) raises -> ShapeN:
    """Validate an activation layer on its per-sample input shape; the output shape is the input's."""
    if code == LAYER_LEAKY_RELU:
        if not isfinite(f):
            raise Error(name + " (LeakyReLU): negative_slope must be a finite number")
    elif code == LAYER_ELU:
        if not isfinite(f):
            raise Error(name + " (ELU): alpha must be a finite number")
    elif code == LAYER_PRELU:
        if a0 < 1 or (a0 != 1 and a0 != cur.d[0]):
            raise Error(name + " (PReLU): num_parameters must be 1 or the input's channels, its first sample axis ("
                        + String(cur.d[0]) + ")")
        var init = prelu_init(a1)
        if not isfinite(init) or not (init > Float32(-1)):
            raise Error(name + " (PReLU): init must be a finite number greater than -1")
    return cur


def nn_act_param_sizes(code: Int, a0: Int) -> List[Int]:
    var out = List[Int]()
    if code == LAYER_PRELU:
        out.append(a0)  # weight (num_parameters)
    return out^


def nn_act_param_names(code: Int) -> List[String]:
    var out = List[String]()
    if code == LAYER_PRELU:
        out.append(".weight")
    return out^


def nn_act_init_bounds(code: Int, a1: Int) -> List[Float32]:
    """model_blocks' init encoding: a negative -v fills v - 1, so PReLU's slope `init` is -(init + 1) (init > -1,
    checked at parse)."""
    var out = List[Float32]()
    if code == LAYER_PRELU:
        out.append(-(prelu_init(a1) + Float32(1)))
    return out^


def nn_act_tensor_dims(code: Int, a0: Int) -> List[Int]:
    """PyTorch's layout of the PReLU weight is params_' own: (num_parameters,)."""
    var out = List[Int]()
    if code == LAYER_PRELU:
        out.append(a0)
    return out^


def _nn_act_param[E: Exec](mut tape: Tape, mut ex: E, P: FP, Gr: FP, off: Int, cols: Int, train: Bool) raises -> Tensor:
    # training: the gradient lands straight in its slot of Gr (model_blocks' _p)
    if train:
        return tape.variable_into(ex, P + off, Gr + off, 1, cols)
    return tape.variable(ex, P + off, 1, cols, False)


def nn_act_forward[E: Exec](
    mut tape: Tape, mut ex: E, code: Int, a0: Int, f: Float32, cur: ShapeN, P: FP, Gr: FP, off: Int, h: Tensor,
    train: Bool,
) raises -> Tensor:
    """Record activation layer `code` on h ((B,) + cur)."""
    if code == LAYER_IDENTITY:
        return h
    if code == LAYER_LEAKY_RELU:
        return imp_unary(tape, ex, h, IMP_LEAKY_RELU, f)
    if code == LAYER_ELU:
        return imp_unary(tape, ex, h, IMP_ELU, f)
    if code == LAYER_MISH:
        return imp_unary(tape, ex, h, IMP_MISH)
    if code == LAYER_HARDSWISH:
        return imp_unary(tape, ex, h, IMP_HARD_SWISH, NN_ACT_SIXTH, NN_ACT_HALF)
    if code == LAYER_HARDSIGMOID:
        return imp_unary(tape, ex, h, IMP_HARD_SIGMOID, NN_ACT_SIXTH, NN_ACT_HALF)
    if code == LAYER_PRELU:
        var w = _nn_act_param(tape, ex, P, Gr, off, a0, train)
        # the slope on the sample's first axis: (C, 1, .., 1) right-aligned to (B,) + cur; shared: (1,)
        var wd = List[Int]()
        wd.append(a0)
        if a0 > 1:
            for _ in range(1, cur.ndim):  # small-loop(ndim: the sample rank, at most 7): shape metadata, no data
                wd.append(1)
        var wv = tape.view_nd(ex, w, rn_shape(wd, "model (PReLU)"))
        return imp_binary(tape, ex, h, wv, IMP_PRELU)
    raise Error("model: unknown activation code " + String(code))
