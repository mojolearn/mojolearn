# SPDX-License-Identifier: Apache-2.0
"""The small MLP's tuned forward and backward as ONE executor op
(lane r1-restore-measured-winners, 2026-11-09; restored from
training/mlp_ops.mojo `mlp_train_step_host` and autograd/rules_small_mlp.mojo
at f59d913b0b, which lane x2 had deleted in favour of the generic
transpose + OPS_LINEAR capture).

    logits = small_mlp(x (rows, in), w1 (hid, in), b1 (hid), w2 (out, hid), b2 (out))
             OPS_SMALL_MLP

    forward   pre1 = x . w1^T (OP_NT), act = relu(pre1 + b1),
              pre2 = act . w2^T (OP_NT), logits = pre2 + b2
    backward  (the hand chain's seven launches, in the tape order the old
              rules launched them; upstream = dL/dlogits)
              db2 = sum_rows(upstream)                 cells op 3
              dw2 = upstream^T . act                   GEMM (out, hid, rows) OP_TN, weight-gradient role
              incoming = upstream . w2                 GEMM (rows, hid, out) OP_NN
              dhidden = relu'(act) incoming            cells op 2
              db1 = sum_rows(dhidden)                  cells op 3
              dw1 = dhidden^T . x                      GEMM (hid, in, rows) OP_TN, weight-gradient role
              dx = dhidden . w1                        GEMM (rows, in, hid) OP_NN, only when x needs a gradient

The GEMMs are the old step's own dispatch (`_gemm`): under FAST the vendor
kernel first (DEVIATION 1876), else, in IDENTICAL, the NN03 neural profile
body at leaf 128 (`neural_model_leaf_into[128]`, SMALL_MLP_GEMM_LEAF128
below), else `identical_gemm_into` with the op's workspace. The elementwise
launches are `_mlp_kernel` at its old geometry (128 threads a block, one
thread per cell, one per column for the row sum), its body now
autograd/small_mlp_cells.mojo so the host arm (autograd/small_mlp_host.mojo,
reached through `Exec.small_mlp_host`) computes the same words.

The op is recorded as ONE tape custom node (SMALL_MLP_TAG) over
[x, w1, b1, w2, b2]; its VJP is reached through autograd/rules_step_ops.mojo
StepOpRules. Saved: act, the two backward scratch rows (incoming,
dhidden), the GEMM workspace, the four dimensions and whether x wants a
gradient. The op keys on nothing but its operands' shapes; the small MLP
program (autograd/small_mlp_program.mojo) is its one capturer, chosen by
MODEL KIND, never by a dimension.

A/B: `-D MOJOLEARN_SMALL_MLP_GENERIC_OPS` makes the program capture the
generic transpose + OPS_LINEAR ops instead (lane x2's form), for the box
comparison mlp-train-step / mlp-predict.
"""
from max.gpu import block_dim, block_idx, thread_idx
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext
from sequence.ops import FP
from sequence.exec_device import DeviceExec
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from autograd.block_step_ops import block_buffer_view
from autograd.small_mlp_cells import (
    SMALL_MLP_BIAS, SMALL_MLP_BIAS_RELU, SMALL_MLP_RELU_BACKWARD, SMALL_MLP_SUM_ROWS,
    small_mlp_cell, small_mlp_cell_count,
)
from gemm.checks.gemm_identical import _fast_vendor_gemm
from gemm.neural_dispatch import (
    identical_gemm_into,
    identical_gemm_workspace_max_floats,
    neural_model_leaf_into,
    NEURAL_GEMM_EXPERIMENT_ENABLED,
)
from gemm.contract import OP_NN, OP_NT, OP_TN
from gemm.experiments.neural_switches import ROLE_PROJECTION, ROLE_WGRAD
from gemm.experiments.neural_profile import NEURAL_EXPERIMENTS_ALLOWED

#: The session op code (autograd/ops_api.mojo) and the tape custom tag
#: (99-104 graph, 108-111 block steps, 112-114 MoE, 115 lm_head_v2,
#: 128+ rank-N; 116 is this op's).
comptime OPS_SMALL_MLP = 88
comptime SMALL_MLP_TAG = 116

# gemm_leaf=neural128 for THE SMALL MLP ONLY (lane grid-act-5, 2026-10-08;
# IDENTICAL grid ge123e6f9, docs/apple-fast/EXPERIMENTS.md). The
# build-wide arm (-D MOJOLEARN_IDN_GEMM_LEAF=1) was a SPLIT: mlp-train-step
# FASTER, NV 2.0 -> 1.7 ms / AMD 3.5 -> 3.1 ms (0.886x combined), but every
# other neural model SLOWER: lm-forward 7.2x, lm-train-step 13.6x,
# mamba1/2/3-forward 3.0/3.8/5.2x, samba-train-step 1.73x,
# transformer-forward 7.6x. Output hashes equal to the incumbent on both
# vendors (no bits change: at leaf 128 the profile's partition is the
# contract's, contract_leaf_size). So the choice is keyed on the MODEL KIND
# (this op's GEMMs are the small MLP's and nobody else's), never on a
# dimension. -D MOJOLEARN_IDN_MLP_GEMM_LEAF128_OFF restores the incumbent
# dispatch here. Any build-wide neural GEMM arm (gemm_leaf=neural128|neural256,
# a schedule, chains, Ozaki) routes the op through identical_gemm_into like
# every other caller, so the grid's gemm_leaf arms keep their meaning.
# (Comment and condition carried from training/mlp_ops.mojo MLP_GEMM_LEAF128
# at f59d913b0b.)
comptime SMALL_MLP_GEMM_LEAF128 = (
    NEURAL_EXPERIMENTS_ALLOWED
    and not NEURAL_GEMM_EXPERIMENT_ENABLED
    and not is_defined["MOJOLEARN_IDN_MLP_GEMM_LEAF128_OFF"]()
)

comptime _P = MutPointer[Float32, MutAnyOrigin]
comptime _TPB = 128


@always_inline
def _p(p: FP) -> _P:
    return _P(unsafe_from_address=Int(p))


def small_mlp_check(rows: Int, n_in: Int, hid: Int, n_out: Int) raises:
    if rows < 1 or n_in < 1 or hid < 1 or n_out < 1:
        raise Error("ValueError: ops: small_mlp needs rows, inputs, hidden and outputs >= 1")
    var widest = max(n_in, max(hid, n_out))
    if hid > 2147483647 // n_in or n_out > 2147483647 // hid or rows > 2147483647 // widest:
        raise Error("ValueError: ops: small_mlp exceeds signed 32-bit kernel indexing")


def _max_ws(mut ws_n: Int, m: Int, n: Int, k: Int):
    comptime if SMALL_MLP_GEMM_LEAF128:
        # The profile body reads no workspace (the measured arm sized it 1).
        return
    var w = identical_gemm_workspace_max_floats(m, n, k)
    if w > ws_n:
        ws_n = w


def small_mlp_workspace_floats(rows: Int, n_in: Int, hid: Int, n_out: Int, backward: Bool) -> Int:
    """One workspace for every GEMM of the op (forward, or forward and
    backward): the max of the certified sizer over the shapes, as the old
    step sized it. The GEMMs run one after another on the one in-order
    context."""
    var ws_n = 1
    _max_ws(ws_n, rows, hid, n_in)
    _max_ws(ws_n, rows, n_out, hid)
    if backward:
        _max_ws(ws_n, n_out, hid, rows)
        _max_ws(ws_n, rows, hid, n_out)
        _max_ws(ws_n, hid, n_in, rows)
        _max_ws(ws_n, rows, n_in, hid)
    return ws_n


def _cells_kernel(source: _P, other: _P, dst: _P, rows_in: Int32, cols_in: Int32, operation_in: Int32):
    """`_mlp_kernel` (f59d913b0b): one thread per item of small_mlp_cell."""
    var rows = Int(rows_in)
    var cols = Int(cols_in)
    var operation = Int(operation_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= small_mlp_cell_count(rows, cols, operation):
        return
    small_mlp_cell(source, other, dst, rows, cols, operation, i)


def _cells(ctx: DeviceContext, source: FP, other: FP, dst: FP, rows: Int, cols: Int, operation: Int) raises:
    """`_launch_mlp`: 128 threads a block, one thread per output item."""
    var count = small_mlp_cell_count(rows, cols, operation)
    ctx.enqueue_function[_cells_kernel](
        _p(source), _p(other), _p(dst), Int32(rows), Int32(cols), Int32(operation),
        grid_dim=((count + _TPB - 1) // _TPB, 1, 1), block_dim=(_TPB, 1, 1),
    )


def _gemm[ROLE: Int = ROLE_PROJECTION](
    ex: DeviceExec, c: FP, a: FP, b: FP, ws: FP, ws_n: Int, m: Int, n: Int, k: Int, op: Int,
) raises:
    """The old step's `_gemm` (f59d913b0b) over executor storage views:
    under FAST the vendor kernel first (DEVIATION 1876), else the leaf-128
    profile body (SMALL_MLP_GEMM_LEAF128), else identical_gemm_into with
    the op's workspace. No wait."""
    var ctx = ex.ctx.copy()
    var cv = block_buffer_view(ex, c, m * n)
    var av = block_buffer_view(ex, a, m * k)
    var bv = block_buffer_view(ex, b, n * k)
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        if _fast_vendor_gemm(ctx, cv, av, bv, m, n, k, op):
            return
    comptime if SMALL_MLP_GEMM_LEAF128:
        # ROLE is only a schedule-arm mask and no schedule arm is on in
        # this build.
        neural_model_leaf_into[128](ctx, cv, av, bv, m, n, k, op)
        return
    var wv = block_buffer_view(ex, ws, ws_n)
    identical_gemm_into[ROLE=ROLE](ctx, cv, av, bv, wv, m, n, k, op)
    _ = wv^


def small_mlp_forward_device(ex: DeviceExec, bufs: List[FP], dims: List[Int]) raises:
    """bufs [x, w1, b1, w2, b2, pre1, act, pre2, logits, ws], dims [rows,
    in, hid, out, ws_n]: the old step's forward, four launches, no wait."""
    if len(bufs) != 10 or len(dims) != 5:
        raise Error("ops: small_mlp forward takes ten buffers and five dimensions")
    var rows = dims[0]
    var n_in = dims[1]
    var hid = dims[2]
    var n_out = dims[3]
    var ws_n = dims[4]
    var ctx = ex.ctx.copy()
    _gemm(ex, bufs[5], bufs[0], bufs[1], bufs[9], ws_n, rows, hid, n_in, OP_NT)
    _cells(ctx, bufs[5], bufs[2], bufs[6], rows, hid, SMALL_MLP_BIAS_RELU)
    _gemm(ex, bufs[7], bufs[6], bufs[3], bufs[9], ws_n, rows, n_out, hid, OP_NT)
    _cells(ctx, bufs[7], bufs[4], bufs[8], rows, n_out, SMALL_MLP_BIAS)


def small_mlp_backward_device(ex: DeviceExec, bufs: List[FP], dims: List[Int]) raises:
    """bufs [x, w1, w2, act, upstream, incoming, dhidden, ws, g_x, g_w1,
    g_b1, g_w2, g_b2], dims [rows, in, hid, out, ws_n, want_dx]: the hand
    chain's seven launches (header), each writing its whole destination."""
    if len(bufs) != 13 or len(dims) != 6:
        raise Error("ops: small_mlp backward takes thirteen buffers and six dimensions")
    var rows = dims[0]
    var n_in = dims[1]
    var hid = dims[2]
    var n_out = dims[3]
    var ws_n = dims[4]
    var ctx = ex.ctx.copy()
    var x = bufs[0]
    var w1 = bufs[1]
    var w2 = bufs[2]
    var act = bufs[3]
    var up = bufs[4]
    var incoming = bufs[5]
    var dhidden = bufs[6]
    var ws = bufs[7]
    # T1 = H2: db2 (its `other` operand is never read)
    _cells(ctx, up, up, bufs[12], rows, n_out, SMALL_MLP_SUM_ROWS)
    # T2 = H1: dw2
    _gemm[ROLE_WGRAD](ex, bufs[11], up, act, ws, ws_n, n_out, hid, rows, OP_TN)
    # T3 = H3: incoming
    _gemm(ex, incoming, up, w2, ws, ws_n, rows, hid, n_out, OP_NN)
    # T4 = H4: dhidden = relu'(act) incoming
    _cells(ctx, act, incoming, dhidden, rows, hid, SMALL_MLP_RELU_BACKWARD)
    # T5 = H6: db1
    _cells(ctx, dhidden, incoming, bufs[10], rows, hid, SMALL_MLP_SUM_ROWS)
    # T6 = H5: dw1
    _gemm[ROLE_WGRAD](ex, bufs[9], dhidden, x, ws, ws_n, hid, n_in, rows, OP_TN)
    # T7 = H7: dx, only when x wants a gradient
    if dims[5] != 0:
        _gemm(ex, bufs[8], dhidden, w1, ws, ws_n, rows, n_in, hid, OP_NN)
