# SPDX-License-Identifier: Apache-2.0
"""The log_softmax layer of a Sequential description (lane nn-losses-2, 2026-11-23): layer code 77.

PyTorch's ``nn.LogSoftmax(dim)``: x - max - log(sum exp(x - max)) along one axis, the numerically stable form. The
layer is a declaration over the EXISTING tape op Tape.log_softmax (OP_AG_SOFTMAX_FWD with its log flag, the row
body autograd/row_ops.mojo row_softmax_fwd, the same body the tape's cross_entropy and softmax use; backward
OP_AG_SOFTMAX_VJP's log form). Along the last axis it is one launch forward and one backward; along another axis
autograd/rankn_api.mojo rn_softmax_along permutes the axis last (a view), the row op materialises the permuted
matrix once (one copy launch), and permutes back (a view). No parameters; nothing is looked up or allocated per
element; the axis is read from the description when the layer is recorded. autograd/model_blocks.mojo dispatches
the code here between its `# nn-losses-2` markers.

  code  layer                     tape op                                launches forward / backward
  77    log_softmax (a0 dim,      Tape.log_softmax via rn_softmax_along   1 / 1 (last axis); 2 / 2 otherwise
        PyTorch's: counted with
        the batch axis; -1 last)
"""
from std.sys.compile import is_defined
from sequence.exec_trait import Exec
from autograd.tape import Tape, Tensor
from autograd.rankn import ShapeN
from autograd.rankn_api import rn_softmax_along

comptime LAYER_LOG_SOFTMAX = 77

# Fault injection (verify's sabotage arm of this family, never in a product build): the layer records the softmax
# (probabilities) instead of its log. checks/native_tests/nn_losses_2.mojo's float64 oracle must FAIL under it.
comptime NN_LOG_SOFTMAX_SABOTAGE = is_defined["MOJOLEARN_NN_LOSSES_SABOTAGE_LOG_SOFTMAX"]()


def is_nn_log_softmax(code: Int) -> Bool:
    return code == LAYER_LOG_SOFTMAX


def log_softmax_axis(dim: Int, cur: ShapeN) -> Int:
    """The sample axis (0-based, without the batch) of PyTorch's `dim` on (B,) + cur: a negative dim counts from
    the last axis, a positive one counts the batch axis as 0. -1 when dim names the batch axis or no axis."""
    var nd = cur.ndim
    if dim < 0:
        var a = nd + dim
        return a if a >= 0 else -1
    if dim == 0 or dim > nd:
        return -1
    return dim - 1


def nn_log_softmax_shape(code: Int, a0: Int, cur: ShapeN, name: String) raises -> ShapeN:
    """Validate the layer on its per-sample input shape; the output shape is the input's."""
    if log_softmax_axis(a0, cur) < 0:
        if a0 == 0:
            raise Error(name + " (LogSoftmax): dim 0 is the batch axis; give an axis of the sample (1 to "
                        + String(cur.ndim) + ", or -1 for the last)")
        raise Error(name + " (LogSoftmax): dim " + String(a0) + " is not an axis of the (batch,) + "
                    + String(cur.ndim) + "-axis input")
    if cur.d[log_softmax_axis(a0, cur)] < 1:
        raise Error(name + " (LogSoftmax): the axis is empty")
    return cur


def nn_log_softmax_forward[E: Exec](
    mut tape: Tape, mut ex: E, a0: Int, cur: ShapeN, h: Tensor,
) raises -> Tensor:
    """Record the layer on h ((B,) + cur): log_softmax along sample axis log_softmax_axis(a0) (h's axis + 1)."""
    var axis = log_softmax_axis(a0, cur) + 1
    comptime if NN_LOG_SOFTMAX_SABOTAGE:
        return rn_softmax_along(tape, ex, h, axis, False)
    return rn_softmax_along(tape, ex, h, axis, True)
