# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The recurrent layer's options (lane nn-recurrent, 2026-11-23): the reverse direction and the final state.

Sequential's `lstm` / `gru` / `rnn` take PyTorch's `num_layers`, `dropout`, `bidirectional` and an `output`
choice. The description pack (bindings/sequential_description_pack.mojo) writes them as existing layers: one
recurrent layer (autograd/model_blocks.mojo LAYER_RECURRENT) per layer and direction, a branch and a concat
joining the two directions on the features, and a dropout layer between stacked layers. What the recurrent
layer itself gains is two description arguments:

  a2  reverse  1: the layer walks time from T - 1 down to 0, each row from its own last valid step
  a3  final    1: the layer's output is its final state (1, hidden) instead of every step's state (T, hidden)

Both are 0 in every description written before this lane, and a layer with both 0 records exactly the launches
it recorded before.

The reverse direction is a DIRECTION of the existing recurrence, not a second implementation: the scan
(sequence/recurrent_scan.mojo fwd_step, autograd/kernels/recurrent_scan.mojo bwd_step) and the per-step chain
(autograd/rules_recurrent.mojo rnn_layer, _layer_bwd) run the same cell bodies on the same launches, with every
step pointer walking back one row block a step (the scan's i4: 1 forward-scan, a negative dGH offset backward).
The states are stored in time order, the zero state after them, so the output needs no permutation. With
lengths the mask's position is the time: a row's padded times (walked first) hold its zero state, then its own
steps run from its last valid one down to 0, as PyTorch's reverse direction over a packed sequence; its output
past the row's length is 0 (PyTorch's padding value), while the forward direction's repeats its final state.

Launches per recurrent layer and training step: reverse = forward, forward and backward, with and without lengths
(the input projection, the scan or the per-step chain, the weight-gradient folds). The final state is a view of
the output's rows at time T - 1 (forward) or 0 (reverse): no launch.
"""
from std.sys.compile import is_defined
from sequence.exec_trait import Exec
from sequence.ops import Args, FP
from autograd.model_ops import OP_AGM_PAD_MASK, OP_AGM_PAD_MASK_BT
from autograd.tape import Tape, Tensor
from autograd.rankn import ShapeN
from autograd.rankn_api import rn_shape
from autograd.rankn_layout import rankn_narrow

#: Fault injection (lane nn-recurrent): the reverse direction ignores the rows' lengths and walks all T steps,
#: so a short row's reverse pass starts in its padding. checks/native_tests/nn_recurrent.mojo must fail.
comptime NN_RECURRENT_SABOTAGE_REVERSE = is_defined["MOJOLEARN_NN_RECURRENT_SABOTAGE_REVERSE"]()


def nn_rnn_reverse_lengths(lengths: Optional[FP]) -> Optional[FP]:
    """The lengths the reverse direction's scan reads: the rows' own (the sabotage build drops them)."""
    comptime if NN_RECURRENT_SABOTAGE_REVERSE:
        return None
    return lengths


def nn_rnn_pad_mask[E: Exec, time_major: Bool](
    mut tape: Tape, mut ex: E, lengths: FP, T: Int, B: Int,
) raises -> Tensor:
    """One float32 padding flag per step and row, in the calling layer's layout."""
    var mask = tape.reserve(ex, T * B)
    var cfg = Args()
    cfg.p0 = lengths
    cfg.p1 = mask
    cfg.i0 = B if time_major else T
    comptime if time_major:
        ex.launch[OP_AGM_PAD_MASK](cfg, T * B)
        return tape.variable(ex, mask, T, B, False)
    else:
        ex.launch[OP_AGM_PAD_MASK_BT](cfg, T * B)
        return tape.variable(ex, mask, B, T, False)


def nn_rnn_final[E: Exec](mut tape: Tape, mut ex: E, y: Tensor, T: Int, B: Int, H: Int, at: Int) raises -> Tensor:
    """The final state (B, 1, H) of a recurrent output y (T B, H, time order): its rows of time `at`."""
    var y3 = tape.view_nd(ex, y, ShapeN.matrix(T, B * H))
    var last = rankn_narrow(tape, ex, y3, 0, at, 1)
    tape.materialize(ex, last)
    var d: List[Int] = [B, 1, H]
    return tape.view_nd(ex, last, rn_shape(d, "recurrent final state"))
