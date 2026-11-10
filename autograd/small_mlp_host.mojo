# SPDX-License-Identifier: Apache-2.0
"""OPS_SMALL_MLP's host arm (autograd/small_mlp_op.mojo has the device
kernels and the op's contract). Only the host executor
(sequence/exec.mojo `HostExec.small_mlp_host`) imports it, so no GPU module
imports host threads.

Same words as the device in IDENTICAL: every GEMM cell is the NN03 profile
cell at leaf 128 (`neural_cell` over `neural_partition[128]`, the body the
device's `neural_model_leaf_into[128]` launches one thread per cell; at
leaf 128 that partition is the contract's, so the same words as the
incumbent dispatch the `MOJOLEARN_IDN_MLP_GEMM_LEAF128_OFF` arm takes), and
every elementwise item is `small_mlp_cell`, the device kernel's body. Rows
(or columns, for the row sums) are split across cores; the fold inside each
cell is unchanged, so the task count moves no bit.
"""
from sequence.ops import FP
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk
from gemm.contract import OP_NN, OP_NT, OP_TN
from gemm.experiments.neural_profile import (
    NEURAL_CHAINS, neural_cell, neural_partition, neural_strides, neural_validate,
)
from autograd.small_mlp_cells import (
    SMALL_MLP_BIAS, SMALL_MLP_BIAS_RELU, SMALL_MLP_RELU_BACKWARD, SMALL_MLP_SUM_ROWS,
    small_mlp_cell, small_mlp_cell_count,
)

comptime _P = MutPointer[Float32, MutAnyOrigin]


@always_inline
def _p(p: FP) -> _P:
    return _P(unsafe_from_address=Int(p))


def _gemm_host(c: FP, a: FP, b: FP, m: Int, n: Int, k: Int, op: Int) raises:
    """C (m, n) = op(A, B), each cell the leaf-128 profile cell; rows of C
    split across cores."""
    neural_validate(m, n, k, op)
    if m == 0 or n == 0:
        return
    var part = neural_partition[128](k)
    var st = neural_strides(op, m, n, k)
    var leaf = part[0]
    var leaves = part[1]
    var asi = st[0]
    var asp = st[1]
    var bsp = st[2]
    var bsj = st[3]
    var cp = _p(c)
    var ap = _p(a)
    var bp = _p(b)
    var tasks = host_predict_task_count_work(m, m * n * max(k, 1))
    var chunk = host_predict_chunk(m, tasks)

    def _rows(t: Int) {imm cp, imm ap, imm bp, imm chunk, imm m, imm n, imm k, imm leaf, imm leaves, imm asi, imm asp, imm bsp, imm bsj}:
        var lo = t * chunk
        for row in range(lo, min(lo + chunk, m)):
            for col in range(n):
                cp.unsafe_store(row * n + col, neural_cell[NEURAL_CHAINS](
                    ap, bp, row, col, k, leaf, leaves, asi, asp, bsp, bsj))

    host_parallelize(_rows, tasks)


def _cells_host(source: FP, other: FP, dst: FP, rows: Int, cols: Int, operation: Int) raises:
    """small_mlp_cell over every item, items split across cores."""
    var count = small_mlp_cell_count(rows, cols, operation)
    var per_item = rows if operation == SMALL_MLP_SUM_ROWS else 1
    var sp = _p(source)
    var op_ = _p(other)
    var dp = _p(dst)
    var tasks = host_predict_task_count_work(count, count * per_item)
    var chunk = host_predict_chunk(count, tasks)

    def _items(t: Int) {imm sp, imm op_, imm dp, imm chunk, imm count, imm rows, imm cols, imm operation}:
        var lo = t * chunk
        for i in range(lo, min(lo + chunk, count)):
            small_mlp_cell(sp, op_, dp, rows, cols, operation, i)

    host_parallelize(_items, tasks)


def small_mlp_host_run(backward: Bool, bufs: List[FP], dims: List[Int]) raises:
    """`Exec.small_mlp_host`: the device functions' buffer and dimension
    lists (autograd/small_mlp_op.mojo small_mlp_forward_device /
    small_mlp_backward_device), the same launches in the same order."""
    if not backward:
        if len(bufs) != 10 or len(dims) != 5:
            raise Error("ops: small_mlp host forward takes ten buffers and five dimensions")
        var rows = dims[0]
        var n_in = dims[1]
        var hid = dims[2]
        var n_out = dims[3]
        _gemm_host(bufs[5], bufs[0], bufs[1], rows, hid, n_in, OP_NT)
        _cells_host(bufs[5], bufs[2], bufs[6], rows, hid, SMALL_MLP_BIAS_RELU)
        _gemm_host(bufs[7], bufs[6], bufs[3], rows, n_out, hid, OP_NT)
        _cells_host(bufs[7], bufs[4], bufs[8], rows, n_out, SMALL_MLP_BIAS)
        return
    if len(bufs) != 13 or len(dims) != 6:
        raise Error("ops: small_mlp host backward takes thirteen buffers and six dimensions")
    var rows = dims[0]
    var n_in = dims[1]
    var hid = dims[2]
    var n_out = dims[3]
    var x = bufs[0]
    var w1 = bufs[1]
    var w2 = bufs[2]
    var act = bufs[3]
    var up = bufs[4]
    var incoming = bufs[5]
    var dhidden = bufs[6]
    _cells_host(up, up, bufs[12], rows, n_out, SMALL_MLP_SUM_ROWS)
    _gemm_host(bufs[11], up, act, n_out, hid, rows, OP_TN)
    _gemm_host(incoming, up, w2, rows, hid, n_out, OP_NN)
    _cells_host(act, incoming, dhidden, rows, hid, SMALL_MLP_RELU_BACKWARD)
    _cells_host(dhidden, incoming, bufs[10], rows, hid, SMALL_MLP_SUM_ROWS)
    _gemm_host(bufs[9], dhidden, x, hid, n_in, rows, OP_TN)
    if dims[5] != 0:
        _gemm_host(bufs[8], dhidden, w1, rows, n_in, hid, OP_NN)
