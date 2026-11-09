# SPDX-License-Identifier: Apache-2.0
# GCNConv and SAGEConv backward on the tape (lane x9-moe-gcn-rules, 2026-11-09; plan row X9,
# docs/plans/EXECUTOR_CONSOLIDATION_AND_CLEANUP.md). The resident form of both layers (the GPU binding,
# bindings/xcnn_graph_native.mojo `_gcn_backward_dev` / `_sage_backward_dev`) records its forward here and runs
# one backward in ONE native call (`x_cnn_gcn_backward_t` / `x_cnn_sage_backward_t`); the door's hand chain of
# four to nine entry crossings is deleted (TOMBSTONE in the door). Lane m1-followups-misc: the host form (every
# address a host array: SAGE project=True, a non-resident FAST form) runs the same entries with the device masks
# 0 (each `_m` entry uploads and downloads its operands, as the deleted host chain's `x_cnn_*` entries did); the
# CPU binding runs the same tape over host rules (autograd/rules_graph_host.mojo). The recording and the address
# words are autograd/graph_tape.mojo.
#
# ---------------------------------------------------------------- the one sparse aggregate op
# GRAPH_AGG is the graph's aggregate y = A h over the cached CSR (fixed row and column order): GCN's
# propagation with its normalised values (plus the bias), SAGE's sum or mean with unit values. Its VJP is
# the transposed fold dh = A^T dy (`spmm_m` over csr_t / vals_t, mode 0); the mean first divides each target
# row by its in-degree read from the forward offsets (`spmm_m` mode 3 over csr_f). GRAPH_MAX is SAGE's max
# aggregate (`graph_op_m` kind 1, the gradient split among tied entries).
#
# ---------------------------------------------------------------- the tape graphs
# GCN:  x (variable_into dx) [n, d], W (variable_into gW) [f, d], [b (variable_into gb) [1, f]]
#       h = GRAPH_DENSE(x, W)      [n, f]  h = x W^T            grad storage dh
#       y = GRAPH_AGG(h[, b])      [n, f]  A h (+ b)            grad storage = the caller's g (the seed, in place)
# SAGE: x (variable_into dx), W_l, b_l (variable_into dW_l, db_l), [W_r (variable_into gW_r)]
#       a = GRAPH_AGG(x) or GRAPH_MAX(x)       [n, d]   grad storage dagg
#       o = GRAPH_SAGE_OUT(a, W_l, b_l[, x, W_r]) [n, f]  a W_l^T + b_l (+ x W_r^T)
#       [y = GRAPH_L2(o)]                       [n, f]   row L2 normalisation
#
# ---------------------------------------------------------------- accumulation order
# Every rule writes its inputs' whole gradients in place (writes_total everywhere: no fill, no scratch, no
# OP_AG_ACCUMULATE). x has two consumers in a SAGE layer with root_weight: GRAPH_SAGE_OUT (recorded later,
# applied first) writes the root term gr = g W_r into the rules' scratch, and the aggregate's rule (applied
# second) joins dx = gr + dx_agg with the hand path's add (`map2_m[add_at]`, gr first). Every other gradient
# has one consumer.
#
# ---------------------------------------------------------------- launch inventory (unchanged words)
# The deleted hand chain's entries, one for one, with the same arguments: GCN [bias: gemm G^T 1], spmm^T,
# gemm dW, gemm dX; SAGE [norm: graph_op 3], linear_backward, [mean: spmm 3], spmm^T | graph_op 1,
# [root: gemm dW_r, gemm g W_r, add | not resident: download]. The root's two GEMMs now run before the
# aggregate's transposed fold (independent outputs: the same words). The tape launches nothing itself;
# each `_m` entry ends in its own wait, the closing sync finds the queue empty.
from autograd.tape import BackwardRules
from sequence.exec_trait import Exec
from sequence.exec_device import DeviceExec
from sequence.ops import FP, Args
from x_cnn.ops import FP as XFP, add_at, relu_bwd_at
from x_cnn.device import cnn_ctx, gemm_m, spmm_m, graph_op_m, map2_m, res_download
from autograd.kernels.cnn import linear_backward_m
from autograd.graph_tape import (
    GRAPH_AGG, GRAPH_DENSE, GRAPH_MAX, GRAPH_SAGE_OUT, GRAPH_L2, GRAPH_PROJ, LAYER_GCN, LAYER_SAGE,
    AGG_SUM, AGG_MEAN, AGG_MAX, GraphCfg, gcn_cfg, sage_cfg, gcn_record_backward, sage_record_backward,
)


def graph_spmm(vals: Int, h: Int, csr: Int, dst: Int, dev: Int, n: Int, F: Int, nnz: Int, mode: Int) raises:
    """`x_cnn_spmm_m`'s body: dst (n x F) = the CSR fold of h, a = [vals, h, csr, dst]; with no
    entries h stands in for the values (no value is read), as the entry does."""
    if n <= 0 or F <= 0 or nnz < 0 or mode < 0 or mode > 3:
        raise Error("graph backward (autograd): positive n and F, nnz >= 0, mode in {0, 1, 2, 3}")
    var a: List[Int] = [vals, h, csr, dst]
    var d = dev
    if nnz <= 0:
        a[0] = a[1]
        d = (d & ~1) | ((d >> 1) & 1)
    var prm: List[Int32] = [Int32(n), Int32(F), Int32(nnz), Int32(mode)]
    spmm_m(a, d, nnz, n + 1 + 2 * nnz, prm)


struct GraphRules(BackwardRules):
    """The VJPs of a GCNConv or SAGEConv layer's backward on the GPU binding.
    Holds the layer's words (`GraphCfg`: resident device handles, or host
    arrays in the host form); all must outlive the backward."""

    var c: GraphCfg

    def __init__(out self, c: GraphCfg):
        self.c = c

    @always_inline
    def _m(self, mask: Int) -> Int:
        # the resident form's operand mask; the host form hands every entry host arrays
        return mask if self.c.resident != 0 else 0

    def supports(self, tag: Int) -> Bool:
        return tag >= GRAPH_AGG and tag <= GRAPH_PROJ

    def zeroed_destination(self, tag: Int, saved: Args, edge: Int) -> Bool:
        # every entry stores every word of the destinations it is handed
        return False

    def writes_total(self, tag: Int, saved: Args, edge: Int) -> Bool:
        # accumulation order (header): each rule writes the whole gradient in place
        return True

    def apply[E: Exec](
        self, mut ex: E, tag: Int, saved: Args, output: FP, upstream: FP,
        inputs: List[FP], gradients: List[FP], rows: List[Int], cols: List[Int],
    ) raises:
        # Every launch goes on cnn_ctx(), the context `ex` wraps.
        if tag == GRAPH_AGG or tag == GRAPH_MAX:
            if self.c.layer == LAYER_GCN:
                self._gcn_agg(Int(upstream), gradients)
            else:
                self._sage_agg(Int(upstream), inputs, gradients)
        elif tag == GRAPH_DENSE:
            self._dense(Int(upstream), inputs, gradients)
        elif tag == GRAPH_SAGE_OUT:
            self._sage_out(Int(upstream), inputs, gradients)
        elif tag == GRAPH_L2:
            self._l2(Int(output), Int(upstream), gradients)
        elif tag == GRAPH_PROJ:
            self._proj(Int(upstream), inputs, gradients)
        else:
            raise Error("unknown graph backward rule")

    def _gcn_agg(self, g: Int, gradients: List[FP]) raises:
        # inputs [h] or [h, b]; gradients [dh] or [dh, gb]
        var c = self.c
        if c.bias != 0:
            if len(gradients) != 2:
                raise Error("graph backward (autograd): a biased GCN aggregate has two inputs")
            # gb = G^T 1 (the ones resident with the layer, or a host column)
            var a: List[Int] = [g, c.ones, Int(gradients[1])]
            gemm_m(a, self._m(3 if c.ones_dev != 0 else 1), c.f, 1, c.n, 2)
        graph_spmm(c.vals_t, g, c.csr_t, Int(gradients[0]), self._m(15), c.n, c.f, c.nnz, 0)

    def _dense(self, dh: Int, inputs: List[FP], gradients: List[FP]) raises:
        # h = x W^T: inputs [x, W]; gradients [dx, gW]
        var c = self.c
        var aw: List[Int] = [dh, Int(inputs[0]), Int(gradients[1])]
        gemm_m(aw, self._m(3), c.f, c.d, c.n, 2)
        var ax: List[Int] = [dh, Int(inputs[1]), Int(gradients[0])]
        gemm_m(ax, self._m(5 if c.dx_dev != 0 else 1), c.n, c.d, c.f, 0)

    def _l2(self, y: Int, g: Int, gradients: List[FP]) raises:
        # the row L2 normalisation's backward on y, g and the denominators
        var c = self.c
        var a: List[Int] = [y, g, c.den, Int(gradients[0]), c.l2_csr]
        var prm: List[Int32] = [Int32(c.n), Int32(c.f), Int32(0), Int32(0)]
        graph_op_m(a, self._m(15), prm, 3)

    def _sage_out(self, g: Int, inputs: List[FP], gradients: List[FP]) raises:
        # inputs [agg, W_l, b_l(, x, W_r)]; gradients [dagg, dW_l, db_l(, dx, gW_r)]
        var c = self.c
        var a: List[Int] = [
            Int(inputs[0]), Int(inputs[1]), g, Int(gradients[0]), Int(gradients[1]), Int(gradients[2]),
        ]
        linear_backward_m(a, self._m(13), c.n, c.d, c.f)
        if c.root != 0:
            if len(inputs) != 5:
                raise Error("graph backward (autograd): a SAGE output with root weight has five inputs")
            var aw: List[Int] = [g, Int(inputs[3]), Int(gradients[4])]
            gemm_m(aw, self._m(3), c.f, c.d, c.n, 2)
            # the root term of dx, joined by the aggregate's rule (or the projection's)
            var ar: List[Int] = [g, Int(inputs[4]), c.gr]
            gemm_m(ar, self._m(5), c.n, c.d, c.f, 0)

    def _sage_agg(self, dagg: Int, inputs: List[FP], gradients: List[FP]) raises:
        # inputs [x] (or [s], the projection); gradients [dx] (or [dsp]); upstream dagg
        var c = self.c
        var src = dagg
        if c.aggr == AGG_MEAN:
            # each target row over its in-degree, read from the forward CSR offsets
            graph_spmm(c.vals_f, dagg, c.csr_f, c.dsc, self._m(15), c.n, c.d, c.nnz, 3)
            src = c.dsc
        # the fold's destination: s's gradient with a projection (its rule joins dx), dx itself in the
        # host form without a root term, else the scratch dxs
        var dst = c.dxs
        if c.project != 0 or (c.resident == 0 and c.root == 0):
            dst = Int(gradients[0])
        if c.aggr == AGG_MAX:
            var a: List[Int] = [Int(inputs[0]), src, c.maxaux, dst, c.csr_t]
            var prm: List[Int32] = [Int32(c.n), Int32(c.d), Int32(c.nnz), Int32(0)]
            graph_op_m(a, self._m(31), prm, 1)
        else:
            graph_spmm(c.vals_t, src, c.csr_t, dst, self._m(15), c.n, c.d, c.nnz, 0)
        if dst == Int(gradients[0]):
            return
        var dx = Int(gradients[0])
        var count = c.n * c.d
        if c.root != 0:
            var a: List[Int] = [c.gr, dst, dx]
            var prm: List[Int32] = [0, 0, 0]
            map2_m[add_at](a, self._m(7 if c.dx_dev != 0 else 3), count, count, count, prm)
        elif c.dx_dev == 0:
            res_download(dst, XFP(unsafe_from_address=dx), count)
        else:
            raise Error("graph backward (autograd): a resident dx without root weight is the aggregate's own output")

    def _proj(self, dsp: Int, inputs: List[FP], gradients: List[FP]) raises:
        # s = relu(x W_p^T + b_p) (host form only): inputs [x, W_p, b_p]; gradients [dx, dW_p, db_p]; upstream dsp.
        # The relu's mask is its saved input pre_p; dx joins the root term first (gr + dxp), as the add did.
        var c = self.c
        if c.resident != 0:
            raise Error("graph backward (autograd): the projection runs in the host form only")
        var count = c.n * c.d
        var zp: List[Int32] = [0, 0, 0]
        var ar: List[Int] = [c.pre, dsp, c.dpre]
        map2_m[relu_bwd_at](ar, 0, count, count, count, zp)
        var dx = Int(gradients[0])
        var dxl = c.dxp if c.root != 0 else dx
        var al: List[Int] = [Int(inputs[0]), Int(inputs[1]), c.dpre, dxl, Int(gradients[1]), Int(gradients[2])]
        linear_backward_m(al, 0, c.n, c.d, c.d)
        if c.root != 0:
            var aa: List[Int] = [c.gr, dxl, dx]
            map2_m[add_at](aa, 0, count, count, count, zp)


def gcn_backward_autograd(a: List[Int], p: List[Int]) raises:
    """GCNConv's backward on one tape (autograd/graph_tape.mojo `gcn_cfg`
    for the words): the resident form (g, x, h, out, dh, vals_t, csr_t
    resident; W, gW, b, gb host; dx and ones resident or host) or, with
    p[7] == 0, the host form (every address a host array)."""
    var c = gcn_cfg(a, p)
    var ctx = cnn_ctx()
    var ex = DeviceExec(ctx)
    gcn_record_backward(ex, GraphRules(c), c, a)


def sage_backward_autograd(a: List[Int], p: List[Int]) raises:
    """SAGEConv's backward on one tape (autograd/graph_tape.mojo `sage_cfg`
    for the words): the resident form (no projection; 23 addresses, 8
    integers) or the host form (33 addresses, 10 integers: every address a
    host array, the projection allowed)."""
    var c = sage_cfg(a, p)
    var ctx = cnn_ctx()
    var ex = DeviceExec(ctx)
    sage_record_backward(ex, GraphRules(c), c, a, p[5] != 0)
