# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The recurrent layer's options (lane nn-recurrent, 2026-11-23): the reverse direction and the final state.

Sequential's `lstm` / `gru` / `rnn` take PyTorch's `num_layers`, `dropout`, `bidirectional` and an `output`
choice. The description pack (bindings/sequential_description_pack.mojo) writes them as existing layers: one
recurrent layer (autograd/model_blocks.mojo LAYER_RECURRENT) per layer and direction, a branch and a concat
joining the two directions on the features, and a dropout layer between stacked layers. What the recurrent
layer itself gains is two description arguments, read here:

  a2  reverse  1: the layer runs over each row's steps in reverse order, from the row's own last valid step
               (its length) down to step 0, as PyTorch's reverse direction over a packed sequence does
  a3  final    1: the layer's output is its final state (1, hidden) instead of every step's state (T, hidden)

Both are 0 in every description written before this lane, and a layer with both 0 records exactly the launches
it recorded before (block_recurrent takes neither branch).

The reverse direction is the forward scan (rnn_scan_op, the same one launch per layer) between two row
permutations of the time-major (T B, features) view: row t B + b reads row r(t, b) B + b, where
r(t, b) = len_b - 1 - t for t < len_b and r(t, b) = t past the row's length (T without lengths). The map is
its own inverse, so the same index words reverse the input, reverse the output back, and carry both
gradients: one OP_GATHER_ROWS launch each way (sequence/ops.mojo op_gather_rows, the estimators' batch
gather), no arithmetic. Past a row's length the scan's mask holds the state, so the reversed output there is
the reverse direction's final state, as the forward direction's output there repeats its final state.

The final state is the scan's own last step in the scan's order (step T - 1; with lengths the mask has held
each row's state since its last valid step), so in both directions it is a view of the scan's output rows
[(T - 1) B, T B): PyTorch's h_n of the layer.

Launches, per recurrent layer and step of training (B rows, T steps), on top of the forward direction's:
  forward   the index (once per layer: 6 elementwise launches without lengths, 10 with), 2 OP_GATHER_ROWS
  backward  2 OP_GATHER_ROWS
  final     no launch (a narrow view of the scan's output; its gradient is the narrow's)
"""
from std.sys.compile import is_defined
from sequence.exec_trait import Exec
from sequence.ops import FP, Args, OP_FILL, OP_GATHER_ROWS, OP_SEQ_IOTA
from autograd.tape import Tape, Tensor
from autograd.rankn import ShapeN
from autograd.rankn_api import rn_shape
from autograd.rankn_layout import rankn_narrow
from autograd.rankn_fold import rankn_binary
from autograd.ops import ADD, SUBTRACT, MULTIPLY
from autograd.import_api import imp_binary, imp_where
from autograd.import_ops import IMP_GE

#: The custom tape tag of the row permutation (autograd/model_blocks.mojo ModelRules answers it).
comptime NN_RNN_REVERSE_TAG = 3360

#: Fault injection (lane nn-recurrent): the reverse direction ignores the rows' lengths and reverses all T
#: steps, so a short row's reverse pass starts in its padding. checks/native_tests/nn_recurrent.mojo must fail.
comptime NN_RECURRENT_SABOTAGE_REVERSE = is_defined["MOJOLEARN_NN_RECURRENT_SABOTAGE_REVERSE"]()

#: The index words are float32 row numbers: exact below 2^24.
comptime _EXACT_ROWS = 1 << 24


def _const[E: Exec](mut tape: Tape, mut ex: E, value: Float32) raises -> Tensor:
    var p = tape.reserve(ex, 1)
    var a = Args()
    a.p0 = p
    a.f0 = value
    ex.launch[OP_FILL](a, 1)
    return tape.variable(ex, p, 1, 1, False)


def _iota[E: Exec](mut tape: Tape, mut ex: E, rows: Int, cols: Int) raises -> Tensor:
    var p = tape.reserve(ex, rows * cols)
    var a = Args()
    a.p0 = p
    ex.launch[OP_SEQ_IOTA](a, rows * cols)
    return tape.variable(ex, p, rows, cols, False)


def nn_rnn_reverse_index[E: Exec](mut tape: Tape, mut ex: E, T: Int, B: Int, lengths: Optional[FP]) raises -> FP:
    """The T B row numbers (float32 words, time-major t B + b) of the reverse direction's permutation:
    r(t, b) B + b with r(t, b) = len_b - 1 - t for t < len_b, else t (len_b = T without lengths).

    With q = t B + b: r B + b = c_b - q, where c_b = (len_b - 1) B + 2 b, and t >= len_b is q >= e_b,
    where e_b = len_b B + b. Computed on the executor from iota words (no host loop over rows or steps), every
    value an integer below 2^24, so every float32 operation is exact."""
    if (T + 1) * B >= _EXACT_ROWS:
        raise Error("ValueError: a reverse recurrent layer takes fewer than 16777216 steps times rows in one "
                    "batch; use a smaller batch_size")
    var q = _iota(tape, ex, T, B)       # (T, B): t B + b
    var bpos = _iota(tape, ex, 1, B)    # (1, B): b
    var masked = Bool(lengths)
    comptime if NN_RECURRENT_SABOTAGE_REVERSE:
        masked = False
    if masked:
        var lens = tape.variable(ex, lengths.value(), 1, B, False)
        var bc = _const(tape, ex, Float32(B))
        var e = rankn_binary(tape, ex, rankn_binary(tape, ex, lens, bc, MULTIPLY), bpos, ADD)
        var c = rankn_binary(tape, ex, rankn_binary(tape, ex, e, bpos, ADD), bc, SUBTRACT)
        var r = rankn_binary(tape, ex, c, q, SUBTRACT)
        var past = imp_binary(tape, ex, q, e, IMP_GE)
        return tape.value_of(ex, imp_where(tape, ex, past, q, r))
    var k = _const(tape, ex, Float32((T - 1) * B))
    var c0 = rankn_binary(tape, ex, rankn_binary(tape, ex, bpos, bpos, ADD), k, ADD)
    return tape.value_of(ex, rankn_binary(tape, ex, c0, q, SUBTRACT))


def nn_rnn_reverse_rows[E: Exec](mut tape: Tape, mut ex: E, x: Tensor, idx: FP, rows: Int, cols: Int) raises -> Tensor:
    """Rows of x (rows, cols) in the order of the index words: out[r] = x[idx[r]] (one OP_GATHER_ROWS).
    The index is its own inverse, so the gradient is the same gather of the upstream rows."""
    var xv = tape.value_of(ex, x)
    var out = tape.reserve(ex, rows * cols)
    var a = Args()
    a.p0 = xv
    a.p1 = idx
    a.p2 = out
    a.i0 = rows
    a.i1 = cols
    a.i2 = 0
    ex.launch[OP_GATHER_ROWS](a, rows * cols)
    var saved = Args()
    saved.p0 = idx
    saved.i0 = rows
    saved.i1 = cols
    return tape.custom(ex, [x], out, rows, cols, NN_RNN_REVERSE_TAG, saved)


def nn_rnn_final[E: Exec](mut tape: Tape, mut ex: E, y: Tensor, T: Int, B: Int, H: Int) raises -> Tensor:
    """The final state (B, 1, H) of a scan output y (T B, H) in the scan's order: its rows of step T - 1."""
    var y3 = tape.view_nd(ex, y, ShapeN.matrix(T, B * H))
    var last = rankn_narrow(tape, ex, y3, 0, T - 1, 1)
    tape.materialize(ex, last)
    var d: List[Int] = [B, 1, H]
    return tape.view_nd(ex, last, rn_shape(d, "recurrent final state"))


# ---------------------------------------------------------------- backward (ModelRules dispatches here)
def nn_recurrent_supports(tag: Int) -> Bool:
    return tag == NN_RNN_REVERSE_TAG


def nn_recurrent_apply[E: Exec](mut ex: E, saved: Args, upstream: FP, gradients: List[FP]) raises:
    """d x[idx[r]] = d out[r]: every row of x's gradient written once (the permutation), so the
    destination needs no zeros (ModelRules.zeroed_destination answers False)."""
    var a = Args()
    a.p0 = upstream
    a.p1 = saved.p0
    a.p2 = gradients[0]
    a.i0 = saved.i0
    a.i1 = saved.i1
    a.i2 = 0
    ex.launch[OP_GATHER_ROWS](a, saved.i0 * saved.i1)
