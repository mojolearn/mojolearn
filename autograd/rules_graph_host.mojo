# SPDX-License-Identifier: Apache-2.0
# GCNConv and SAGEConv backward on the tape, the CPU binding's rules (lane m1-followups-misc, 2026-11-09; plan row X9
# leftover). The same recording as the GPU binding (autograd/graph_tape.mojo) over host arrays; each rule runs the
# host bodies the deleted host chain's `x_cnn_*` entries ran (bindings/_mojolearn_x_cnn_host.mojo: gemm, spmm,
# graph_op, linear_backward, relu_backward, add), on the same arguments in the same order, including the CPU
# column's output seam `out_f32` after each, so the CPU column's words do not move. No device import.
from autograd.tape import BackwardRules
from sequence.exec_trait import Exec
from sequence.exec import HostExec
from sequence.ops import FP, Args
from core.hostptr import f32_ptr, read_f32, copy_f32
from x_cnn.ops import FP as XFP, add_at, relu_bwd_at
from x_cnn.host.gemm_host import gemm_host_into
from x_cnn.host.ops_host import map2_into, linear_backward_into, spmm_host, graph_op_host
from x_cnn.host.binding_entries_impl import out_f32, csr_checked
from autograd.graph_tape import (
    GRAPH_AGG, GRAPH_DENSE, GRAPH_MAX, GRAPH_SAGE_OUT, GRAPH_L2, GRAPH_PROJ, LAYER_GCN,
    AGG_MEAN, AGG_MAX, GraphCfg, gcn_cfg, sage_cfg, gcn_record_backward, sage_record_backward,
)


@always_inline
def _x(a: Int) -> XFP:
    return XFP(unsafe_from_address=a)


def _gemm(a: Int, b: Int, c: Int, m: Int, n: Int, k: Int, op: Int):
    """`x_cnn_gemm` (host): C (m x n) = op(A, B)."""
    gemm_host_into(_x(a), _x(b), _x(c), op, m, n, k)
    out_f32(_x(c), m * n)


def _spmm(vals: Int, h: Int, csr: Int, dst: Int, n: Int, F: Int, nnz: Int, mode: Int) raises:
    """`x_cnn_spmm` (host): dst (n x F) = the CSR fold of h."""
    var t = csr_checked(csr, n, F, nnz, mode)
    var v = read_f32(vals, nnz) if nnz > 0 else List[Float32](length=1, fill=Float32(0))
    var hv = read_f32(h, n * F)
    var y = spmm_host(v, hv, t[0], t[1])
    copy_f32(y.unsafe_ptr(), f32_ptr(dst), n * F)


def _graph_op(a: Int, b: Int, aux: Int, dst: Int, csr: Int, n: Int, F: Int, nnz: Int, kind: Int) raises:
    """`x_cnn_graph_op` (host), kinds 1 (SAGE max backward) and 3 (row L2 backward)."""
    var t = csr_checked(csr, n, F, nnz, 0)
    var naux = 2 * n * F if kind < 2 else n
    var av = read_f32(a, n * F)
    var bv = read_f32(b, n * F)
    var auxv = read_f32(aux, naux)
    var r = graph_op_host(av, bv, auxv, t[0], t[1], kind)
    copy_f32(r.unsafe_ptr(), f32_ptr(dst), n * F)
    copy_f32(r.unsafe_ptr() + n * F, f32_ptr(aux), naux)


def _linear_backward(x: Int, w: Int, g: Int, dx: Int, dw: Int, db: Int, n: Int, d_in: Int, d_out: Int):
    """`x_cnn_linear_backward` (host)."""
    linear_backward_into(_x(x), _x(w), _x(g), _x(dx), _x(dw), _x(db), n, d_in, d_out)
    out_f32(_x(dx), n * d_in)
    out_f32(_x(dw), d_out * d_in)
    out_f32(_x(db), d_out)


def _add(a: Int, b: Int, dst: Int, count: Int):
    """`x_cnn_add` (host)."""
    map2_into[add_at](_x(a), _x(b), _x(dst), count)
    out_f32(_x(dst), count)


struct GraphHostRules(BackwardRules):
    """The VJPs of a GCNConv or SAGEConv layer's backward on the CPU
    binding: every address a host array (`GraphCfg` with resident 0)."""

    var c: GraphCfg

    def __init__(out self, c: GraphCfg):
        self.c = c

    def supports(self, tag: Int) -> Bool:
        return tag >= GRAPH_AGG and tag <= GRAPH_PROJ

    def zeroed_destination(self, tag: Int, saved: Args, edge: Int) -> Bool:
        # every body stores every word of the destinations it is handed
        return False

    def writes_total(self, tag: Int, saved: Args, edge: Int) -> Bool:
        # accumulation order (autograd/rules_graph.mojo header): each rule writes the whole gradient in place
        return True

    def apply[E: Exec](
        self, mut ex: E, tag: Int, saved: Args, output: FP, upstream: FP,
        inputs: List[FP], gradients: List[FP], rows: List[Int], cols: List[Int],
    ) raises:
        var c = self.c
        var up = Int(upstream)
        if tag == GRAPH_AGG or tag == GRAPH_MAX:
            if c.layer == LAYER_GCN:
                # inputs [h] or [h, b]; gradients [dh] or [dh, gb]; gb = G^T 1 first, as the host chain did
                if c.bias != 0:
                    _gemm(up, c.ones, Int(gradients[1]), c.f, 1, c.n, 2)
                _spmm(c.vals_t, up, c.csr_t, Int(gradients[0]), c.n, c.f, c.nnz, 0)
                return
            # SAGE: inputs [x] or [s]; gradients [dx] or [dsp]
            var src = up
            if c.aggr == AGG_MEAN:
                _spmm(c.vals_f, up, c.csr_f, c.dsc, c.n, c.d, c.nnz, 3)
                src = c.dsc
            var dst = Int(gradients[0]) if (c.project != 0 or c.root == 0) else c.dxs
            if c.aggr == AGG_MAX:
                _graph_op(Int(inputs[0]), src, c.maxaux, dst, c.csr_t, c.n, c.d, c.nnz, 1)
            else:
                _spmm(c.vals_t, src, c.csr_t, dst, c.n, c.d, c.nnz, 0)
            if dst != Int(gradients[0]):
                # the root term first (gr + dx_agg), as the host chain's add did
                _add(c.gr, dst, Int(gradients[0]), c.n * c.d)
        elif tag == GRAPH_DENSE:
            # h = x W^T: inputs [x, W]; gradients [dx, gW]
            _gemm(up, Int(inputs[0]), Int(gradients[1]), c.f, c.d, c.n, 2)
            _gemm(up, Int(inputs[1]), Int(gradients[0]), c.n, c.d, c.f, 0)
        elif tag == GRAPH_SAGE_OUT:
            # inputs [agg, W_l, b_l(, x, W_r)]; gradients [dagg, dW_l, db_l(, dx, gW_r)]
            _linear_backward(
                Int(inputs[0]), Int(inputs[1]), up, Int(gradients[0]), Int(gradients[1]), Int(gradients[2]),
                c.n, c.d, c.f,
            )
            if c.root != 0:
                if len(inputs) != 5:
                    raise Error("graph backward (autograd): a SAGE output with root weight has five inputs")
                _gemm(up, Int(inputs[3]), Int(gradients[4]), c.f, c.d, c.n, 2)
                _gemm(up, Int(inputs[4]), c.gr, c.n, c.d, c.f, 0)
        elif tag == GRAPH_L2:
            _graph_op(Int(output), up, c.den, Int(gradients[0]), c.l2_csr, c.n, c.f, 0, 3)
        elif tag == GRAPH_PROJ:
            # s = relu(x W_p^T + b_p): inputs [x, W_p, b_p]; gradients [dx, dW_p, db_p]
            var count = c.n * c.d
            map2_into[relu_bwd_at](_x(c.pre), _x(up), _x(c.dpre), count)
            out_f32(_x(c.dpre), count)
            var dx = Int(gradients[0])
            var dxl = c.dxp if c.root != 0 else dx
            _linear_backward(Int(inputs[0]), Int(inputs[1]), c.dpre, dxl, Int(gradients[1]), Int(gradients[2]), c.n, c.d, c.d)
            if c.root != 0:
                _add(c.gr, dxl, dx, count)
        else:
            raise Error("unknown graph backward rule")


def gcn_backward_host_autograd(a: List[Int], p: List[Int]) raises:
    """GCNConv's backward on one tape, the CPU binding (the host form: p[7] == 0)."""
    var c = gcn_cfg(a, p)
    if c.resident != 0:
        raise Error("gcn backward (autograd): the CPU binding runs the host form (p[7] == 0)")
    var ex = HostExec()
    gcn_record_backward(ex, GraphHostRules(c), c, a)


def sage_backward_host_autograd(a: List[Int], p: List[Int]) raises:
    """SAGEConv's backward on one tape, the CPU binding (the host form: 33 addresses, 10 integers)."""
    var c = sage_cfg(a, p)
    if c.resident != 0:
        raise Error("sage backward (autograd): the CPU binding runs the host form (p[8] == 0)")
    var ex = HostExec()
    sage_record_backward(ex, GraphHostRules(c), c, a, p[5] != 0)
