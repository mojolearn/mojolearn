# SPDX-License-Identifier: Apache-2.0
# GCNConv and SAGEConv backward on the tape (lane x9-moe-gcn-rules, 2026-11-09; plan row X9,
# docs/plans/EXECUTOR_CONSOLIDATION_AND_CLEANUP.md). The resident form of both layers (the GPU binding,
# bindings/xcnn_graph_native.mojo `_gcn_backward_dev` / `_sage_backward_dev`) records its forward here and runs
# one backward in ONE native call (`x_cnn_gcn_backward_t` / `x_cnn_sage_backward_t`); the door's hand chain of
# four to nine entry crossings is deleted (TOMBSTONE in the door).
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
from autograd.tape import Tape, Tensor, BackwardRules
from sequence.exec_trait import Exec
from sequence.exec_device import DeviceExec
from sequence.ops import FP, Args
from x_cnn.ops import FP as XFP, add_at
from x_cnn.device import cnn_ctx, gemm_m, spmm_m, graph_op_m, map2_m, res_download
from autograd.kernels.cnn import linear_backward_m

# Tags 99-111 are reserved for the per-layer rules (autograd/rules_cnn.mojo); MoE is 112-114.
comptime GRAPH_AGG = 99
comptime GRAPH_DENSE = 100
comptime GRAPH_MAX = 101
comptime GRAPH_SAGE_OUT = 102
comptime GRAPH_L2 = 103

comptime LAYER_GCN = 0
comptime LAYER_SAGE = 1

#: SAGE aggregation kinds (the door's `aggr` word)
comptime AGG_SUM = 0
comptime AGG_MEAN = 1
comptime AGG_MAX = 2


@always_inline
def _t(a: Int) -> FP:
    return FP(unsafe_from_address=a)


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
    """The VJPs of a GCNConv or SAGEConv layer's resident backward. Holds
    the layer's resident handles (device addresses) and host addresses; all
    must outlive the backward."""

    var layer: Int
    var n: Int
    var d: Int
    var f: Int
    var nnz: Int
    var bias: Int
    var dx_dev: Int
    var ones: Int
    var ones_dev: Int
    var aggr: Int
    var root: Int
    var vals_f: Int
    var csr_f: Int
    var vals_t: Int
    var csr_t: Int
    var maxaux: Int
    var dsc: Int
    var dxs: Int
    var gr: Int
    var den: Int

    def __init__(out self):
        self.layer = LAYER_GCN
        self.n = 0
        self.d = 0
        self.f = 0
        self.nnz = 0
        self.bias = 0
        self.dx_dev = 0
        self.ones = 0
        self.ones_dev = 0
        self.aggr = AGG_SUM
        self.root = 0
        self.vals_f = 0
        self.csr_f = 0
        self.vals_t = 0
        self.csr_t = 0
        self.maxaux = 0
        self.dsc = 0
        self.dxs = 0
        self.gr = 0
        self.den = 0

    def supports(self, tag: Int) -> Bool:
        return tag >= GRAPH_AGG and tag <= GRAPH_L2

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
            if self.layer == LAYER_GCN:
                self._gcn_agg(Int(upstream), gradients)
            else:
                self._sage_agg(Int(upstream), inputs, gradients)
        elif tag == GRAPH_DENSE:
            self._dense(Int(upstream), inputs, gradients)
        elif tag == GRAPH_SAGE_OUT:
            self._sage_out(Int(upstream), inputs, gradients)
        elif tag == GRAPH_L2:
            self._l2(Int(output), Int(upstream), gradients)
        else:
            raise Error("unknown graph backward rule")

    def _gcn_agg(self, g: Int, gradients: List[FP]) raises:
        # inputs [h] or [h, b]; gradients [dh] or [dh, gb]
        if self.bias != 0:
            if len(gradients) != 2:
                raise Error("graph backward (autograd): a biased GCN aggregate has two inputs")
            # gb = G^T 1 (the ones resident with the layer, or a host column)
            var a: List[Int] = [g, self.ones, Int(gradients[1])]
            gemm_m(a, 3 if self.ones_dev != 0 else 1, self.f, 1, self.n, 2)
        graph_spmm(self.vals_t, g, self.csr_t, Int(gradients[0]), 15, self.n, self.f, self.nnz, 0)

    def _dense(self, dh: Int, inputs: List[FP], gradients: List[FP]) raises:
        # h = x W^T: inputs [x, W]; gradients [dx, gW]
        var aw: List[Int] = [dh, Int(inputs[0]), Int(gradients[1])]
        gemm_m(aw, 3, self.f, self.d, self.n, 2)
        var ax: List[Int] = [dh, Int(inputs[1]), Int(gradients[0])]
        gemm_m(ax, 5 if self.dx_dev != 0 else 1, self.n, self.d, self.f, 0)

    def _l2(self, y: Int, g: Int, gradients: List[FP]) raises:
        # the row L2 normalisation's backward on the resident y, g and denominators
        var a: List[Int] = [y, g, self.den, Int(gradients[0]), 0]
        var prm: List[Int32] = [Int32(self.n), Int32(self.f), Int32(0), Int32(0)]
        graph_op_m(a, 15, prm, 3)

    def _sage_out(self, g: Int, inputs: List[FP], gradients: List[FP]) raises:
        # inputs [agg, W_l, b_l(, x, W_r)]; gradients [dagg, dW_l, db_l(, dx, gW_r)]
        var a: List[Int] = [
            Int(inputs[0]), Int(inputs[1]), g, Int(gradients[0]), Int(gradients[1]), Int(gradients[2]),
        ]
        linear_backward_m(a, 13, self.n, self.d, self.f)
        if self.root != 0:
            if len(inputs) != 5:
                raise Error("graph backward (autograd): a SAGE output with root weight has five inputs")
            var aw: List[Int] = [g, Int(inputs[3]), Int(gradients[4])]
            gemm_m(aw, 3, self.f, self.d, self.n, 2)
            # the root term of dx, joined by the aggregate's rule
            var ar: List[Int] = [g, Int(inputs[4]), self.gr]
            gemm_m(ar, 5, self.n, self.d, self.f, 0)

    def _sage_agg(self, dagg: Int, inputs: List[FP], gradients: List[FP]) raises:
        # inputs [x]; gradients [dx]; upstream dagg
        var src = dagg
        if self.aggr == AGG_MEAN:
            # each target row over its in-degree, read from the forward CSR offsets
            graph_spmm(self.vals_f, dagg, self.csr_f, self.dsc, 15, self.n, self.d, self.nnz, 3)
            src = self.dsc
        if self.aggr == AGG_MAX:
            var a: List[Int] = [Int(inputs[0]), src, self.maxaux, self.dxs, self.csr_t]
            var prm: List[Int32] = [Int32(self.n), Int32(self.d), Int32(self.nnz), Int32(0)]
            graph_op_m(a, 31, prm, 1)
        else:
            graph_spmm(self.vals_t, src, self.csr_t, self.dxs, 15, self.n, self.d, self.nnz, 0)
        var dx = Int(gradients[0])
        var count = self.n * self.d
        if self.root != 0:
            var a: List[Int] = [self.gr, self.dxs, dx]
            var prm: List[Int32] = [0, 0, 0]
            map2_m[add_at](a, 7 if self.dx_dev != 0 else 3, count, count, count, prm)
        elif self.dx_dev == 0:
            res_download(self.dxs, XFP(unsafe_from_address=dx), count)
        elif self.dxs != dx:
            raise Error("graph backward (autograd): a resident dx without root weight is the aggregate's own output")


def gcn_backward_autograd(a: List[Int], p: List[Int]) raises:
    """GCNConv's resident backward on one tape.

    a = [g, x, h, out, W, gW, dx, dh, vals_t, csr_t, b, gb, ones]: g, x,
    h, out, dh, vals_t, csr_t resident; W, gW, b, gb host; dx resident or
    host (p[5]); ones resident or host (p[6]); b, gb, ones 0 without a
    bias. p = [n, d, f, nnz, bias, dx resident, ones resident]."""
    if len(a) != 13 or len(p) != 7:
        raise Error("gcn backward (autograd): 13 addresses and 7 integers")
    var n = p[0]
    var d = p[1]
    var f = p[2]
    if n <= 0 or d <= 0 or f <= 0 or p[3] < 0:
        raise Error("gcn backward (autograd): positive n, in and out channels, nnz >= 0")
    var r = GraphRules()
    r.layer = LAYER_GCN
    r.n = n
    r.d = d
    r.f = f
    r.nnz = p[3]
    r.bias = p[4]
    r.dx_dev = p[5]
    r.ones = a[12]
    r.ones_dev = p[6]
    r.vals_t = a[8]
    r.csr_t = a[9]
    var ctx = cnn_ctx()
    var ex = DeviceExec(ctx)
    var tape = Tape()
    var x = tape.variable_into(ex, _t(a[1]), _t(a[6]), n, d)
    var w = tape.variable_into(ex, _t(a[4]), _t(a[5]), f, d)
    var h = tape.custom_into(ex, [x, w], _t(a[2]), _t(a[7]), n, f, GRAPH_DENSE, Args())
    var y: Tensor
    if r.bias != 0:
        var b = tape.variable_into(ex, _t(a[10]), _t(a[11]), 1, f)
        y = tape.custom_into(ex, [h, b], _t(a[3]), _t(a[0]), n, f, GRAPH_AGG, Args())
    else:
        y = tape.custom_into(ex, [h], _t(a[3]), _t(a[0]), n, f, GRAPH_AGG, Args())
    var seed = tape.variable(ex, _t(a[0]), n, f, False)
    tape.backward_with_rules(ex, y, seed, r)
    tape.release(ex)
    ex.sync()


def sage_backward_autograd(a: List[Int], p: List[Int]) raises:
    """SAGEConv's resident backward (no projection) on one tape.

    a = [g, x, agg, out, W_l, b_l, dW_l, db_l, dagg, dx, dxs, vals_f,
    csr_f, vals_t, csr_t, maxaux, dsc, W_r, gW_r, gr, pre, gpre, den]:
    W_l, b_l, dW_l, db_l, W_r, gW_r host; dx resident or host (p[7]); the
    rest resident (dxs is dx itself when dx is resident and there is no
    root weight); an address a configuration does not read is 0. With
    normalize, out is the normalised y and pre / gpre / den the
    pre-normalisation output, its gradient and the row denominators.
    p = [n, d, f, nnz, aggr (0 sum, 1 mean, 2 max), normalize, root
    weight, dx resident]."""
    if len(a) != 23 or len(p) != 8:
        raise Error("sage backward (autograd): 23 addresses and 8 integers")
    var n = p[0]
    var d = p[1]
    var f = p[2]
    if n <= 0 or d <= 0 or f <= 0 or p[3] < 0 or p[4] < AGG_SUM or p[4] > AGG_MAX:
        raise Error("sage backward (autograd): positive n, in and out channels, nnz >= 0, aggr in 0..2")
    var norm = p[5] != 0
    var r = GraphRules()
    r.layer = LAYER_SAGE
    r.n = n
    r.d = d
    r.f = f
    r.nnz = p[3]
    r.aggr = p[4]
    r.root = p[6]
    r.dx_dev = p[7]
    r.dxs = a[10]
    r.vals_f = a[11]
    r.csr_f = a[12]
    r.vals_t = a[13]
    r.csr_t = a[14]
    r.maxaux = a[15]
    r.dsc = a[16]
    r.gr = a[19]
    r.den = a[22]
    var ctx = cnn_ctx()
    var ex = DeviceExec(ctx)
    var tape = Tape()
    var x = tape.variable_into(ex, _t(a[1]), _t(a[9]), n, d)
    var wl = tape.variable_into(ex, _t(a[4]), _t(a[6]), f, d)
    var bl = tape.variable_into(ex, _t(a[5]), _t(a[7]), 1, f)
    var agg = tape.custom_into(
        ex, [x], _t(a[2]), _t(a[8]), n, d, GRAPH_MAX if r.aggr == AGG_MAX else GRAPH_AGG, Args(),
    )
    # the layer output before the normalisation: pre / gpre with normalize, else out / g
    var ov = a[20] if norm else a[3]
    var og = a[21] if norm else a[0]
    var o: Tensor
    if r.root != 0:
        var wr = tape.variable_into(ex, _t(a[17]), _t(a[18]), f, d)
        o = tape.custom_into(ex, [agg, wl, bl, x, wr], _t(ov), _t(og), n, f, GRAPH_SAGE_OUT, Args())
    else:
        o = tape.custom_into(ex, [agg, wl, bl], _t(ov), _t(og), n, f, GRAPH_SAGE_OUT, Args())
    var y = o
    if norm:
        y = tape.custom_into(ex, [o], _t(a[3]), _t(a[0]), n, f, GRAPH_L2, Args())
    var seed = tape.variable(ex, _t(a[0]), n, f, False)
    tape.backward_with_rules(ex, y, seed, r)
    tape.release(ex)
    ex.sync()
