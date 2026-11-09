# SPDX-License-Identifier: Apache-2.0
# GCNConv and SAGEConv backward on the tape: the recording and the address words, shared by the device rules
# (autograd/rules_graph.mojo, the GPU binding, resident and host-form) and the host rules
# (autograd/rules_graph_host.mojo, the CPU binding). Lane m1-followups-misc split it out of rules_graph.mojo so the
# CPU binding and the host form (SAGE project=True, a non-resident FAST form) run the same tape entry as the resident
# form; the host-form hand chains in bindings/xcnn_graph_native.mojo (`_gcn_backward` / `_sage_backward` host
# branches) are deleted (TOMBSTONE there). See rules_graph.mojo's header for the tape graphs and the accumulation
# order; the projection adds:
# SAGE project=True (host form only):
#       s = GRAPH_PROJ(x, W_p, b_p)              [n, d]   relu(x W_p^T + b_p), the aggregate's input; grad storage dsp
#       the aggregate then reads s; GRAPH_PROJ's rule writes dx (joins the root term gr first, as the add did).
# No device import here: the CPU binding compiles this file.
from autograd.tape import Tape, Tensor, BackwardRules
from sequence.exec_trait import Exec
from sequence.ops import FP, Args

# Tags 99-111 are reserved for the per-layer rules (autograd/rules_cnn.mojo); MoE is 112-114.
comptime GRAPH_AGG = 99
comptime GRAPH_DENSE = 100
comptime GRAPH_MAX = 101
comptime GRAPH_SAGE_OUT = 102
comptime GRAPH_L2 = 103
comptime GRAPH_PROJ = 104

comptime LAYER_GCN = 0
comptime LAYER_SAGE = 1

# SAGE aggregations (the door's kind: 0 sum / add, 1 mean, 2 max)
comptime AGG_SUM = 0
comptime AGG_MEAN = 1
comptime AGG_MAX = 2

# the address-list lengths: the resident form, and the host form's extended list
comptime GCN_ADDRS = 13
comptime SAGE_ADDRS = 23
comptime SAGE_HOST_ADDRS = 33


@always_inline
def gt_ptr(a: Int) -> FP:
    return FP(unsafe_from_address=a)


struct GraphCfg(ImplicitlyCopyable, Movable):
    """The words of one GCNConv or SAGEConv backward: shapes, flags and the
    addresses the rules read beyond their tape inputs. `resident` 1: the
    forward's buffers are device handles (the GPU binding's resident form);
    0: every address is a host array (the host form)."""

    var layer: Int
    var resident: Int
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
    var project: Int
    var vals_f: Int
    var csr_f: Int
    var vals_t: Int
    var csr_t: Int
    var maxaux: Int
    var dsc: Int
    var dxs: Int
    var gr: Int
    var den: Int
    var l2_csr: Int
    var pre: Int
    var dpre: Int
    var dxp: Int

    def __init__(out self):
        self.layer = LAYER_GCN
        self.resident = 1
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
        self.project = 0
        self.vals_f = 0
        self.csr_f = 0
        self.vals_t = 0
        self.csr_t = 0
        self.maxaux = 0
        self.dsc = 0
        self.dxs = 0
        self.gr = 0
        self.den = 0
        self.l2_csr = 0
        self.pre = 0
        self.dpre = 0
        self.dxp = 0


def gcn_cfg(a: List[Int], p: List[Int]) raises -> GraphCfg:
    """a = [g, x, h, out, W, gW, dx, dh, vals_t, csr_t, b, gb, ones]; p =
    [n, d, f, nnz, bias, dx resident, ones resident(, resident)]: the
    eighth word 0 is the host form (every address a host array; h and out
    are recorded, never read)."""
    if len(a) != GCN_ADDRS or (len(p) != 7 and len(p) != 8):
        raise Error("gcn backward (autograd): 13 addresses and 7 or 8 integers")
    var c = GraphCfg()
    c.layer = LAYER_GCN
    c.n = p[0]
    c.d = p[1]
    c.f = p[2]
    if c.n <= 0 or c.d <= 0 or c.f <= 0 or p[3] < 0:
        raise Error("gcn backward (autograd): positive n, in and out channels, nnz >= 0")
    c.nnz = p[3]
    c.bias = p[4]
    c.dx_dev = p[5]
    c.ones = a[12]
    c.ones_dev = p[6]
    c.resident = p[7] if len(p) == 8 else 1
    if c.resident == 0 and (c.dx_dev != 0 or c.ones_dev != 0):
        raise Error("gcn backward (autograd): the host form has no resident buffer")
    c.vals_t = a[8]
    c.csr_t = a[9]
    return c


def gcn_record_backward[E: Exec, R: BackwardRules](mut ex: E, rules: R, c: GraphCfg, a: List[Int]) raises:
    """Record GCNConv's forward over the saved buffers and run one tape
    backward with `rules` (the rules read `c`)."""
    var n = c.n
    var d = c.d
    var f = c.f
    var tape = Tape()
    var x = tape.variable_into(ex, gt_ptr(a[1]), gt_ptr(a[6]), n, d)
    var w = tape.variable_into(ex, gt_ptr(a[4]), gt_ptr(a[5]), f, d)
    var h = tape.custom_into(ex, [x, w], gt_ptr(a[2]), gt_ptr(a[7]), n, f, GRAPH_DENSE, Args())
    var y: Tensor
    if c.bias != 0:
        var b = tape.variable_into(ex, gt_ptr(a[10]), gt_ptr(a[11]), 1, f)
        y = tape.custom_into(ex, [h, b], gt_ptr(a[3]), gt_ptr(a[0]), n, f, GRAPH_AGG, Args())
    else:
        y = tape.custom_into(ex, [h], gt_ptr(a[3]), gt_ptr(a[0]), n, f, GRAPH_AGG, Args())
    var seed = tape.variable(ex, gt_ptr(a[0]), n, f, False)
    tape.backward_with_rules(ex, y, seed, rules)
    tape.release(ex)
    ex.sync()


def sage_cfg(a: List[Int], p: List[Int]) raises -> GraphCfg:
    """a = [g, x, agg, out, W_l, b_l, dW_l, db_l, dagg, dx, dxs, vals_f,
    csr_f, vals_t, csr_t, maxaux, dsc, W_r, gW_r, gr, pre, gpre, den] (the
    resident form, 23 addresses) plus, in the host form (33 addresses), [s
    (the projected x, relu output), pre_p (the projection's linear output),
    W_p, b_p, dW_p, db_p, dsp (s's gradient), dpre_p, l2 csr (n + 1 zero
    offsets), dxp (the projection's dx before the root join)]; an address a
    configuration does not read is 0. p = [n, d, f, nnz, aggr (0 sum, 1
    mean, 2 max), normalize, root weight, dx resident(, resident,
    project)]."""
    if not ((len(a) == SAGE_ADDRS and len(p) == 8) or (len(a) == SAGE_HOST_ADDRS and len(p) == 10)):
        raise Error("sage backward (autograd): 23 addresses and 8 integers, or 33 and 10")
    var c = GraphCfg()
    c.layer = LAYER_SAGE
    c.n = p[0]
    c.d = p[1]
    c.f = p[2]
    if c.n <= 0 or c.d <= 0 or c.f <= 0 or p[3] < 0 or p[4] < AGG_SUM or p[4] > AGG_MAX:
        raise Error("sage backward (autograd): positive n, in and out channels, nnz >= 0, aggr in 0..2")
    c.nnz = p[3]
    c.aggr = p[4]
    c.root = p[6]
    c.dx_dev = p[7]
    c.dxs = a[10]
    c.vals_f = a[11]
    c.csr_f = a[12]
    c.vals_t = a[13]
    c.csr_t = a[14]
    c.maxaux = a[15]
    c.dsc = a[16]
    c.gr = a[19]
    c.den = a[22]
    if len(p) == 10:
        c.resident = p[8]
        c.project = p[9]
        c.pre = a[24]
        c.dpre = a[30]
        c.l2_csr = a[31]
        c.dxp = a[32]
    if c.resident != 0 and c.project != 0:
        raise Error("sage backward (autograd): the projection runs in the host form only")
    if c.resident == 0 and c.dx_dev != 0:
        raise Error("sage backward (autograd): the host form has no resident buffer")
    return c


def sage_record_backward[E: Exec, R: BackwardRules](mut ex: E, rules: R, c: GraphCfg, a: List[Int], normalize: Bool) raises:
    """Record SAGEConv's forward over the saved buffers and run one tape
    backward with `rules` (the rules read `c`)."""
    var n = c.n
    var d = c.d
    var f = c.f
    var tape = Tape()
    var x = tape.variable_into(ex, gt_ptr(a[1]), gt_ptr(a[9]), n, d)
    var wl = tape.variable_into(ex, gt_ptr(a[4]), gt_ptr(a[6]), f, d)
    var bl = tape.variable_into(ex, gt_ptr(a[5]), gt_ptr(a[7]), 1, f)
    # the aggregate's input: x, or the projection s = relu(x W_p^T + b_p)
    var src = x
    if c.project != 0:
        var wp = tape.variable_into(ex, gt_ptr(a[25]), gt_ptr(a[27]), d, d)
        var bp = tape.variable_into(ex, gt_ptr(a[26]), gt_ptr(a[28]), 1, d)
        src = tape.custom_into(ex, [x, wp, bp], gt_ptr(a[23]), gt_ptr(a[29]), n, d, GRAPH_PROJ, Args())
    var agg = tape.custom_into(
        ex, [src], gt_ptr(a[2]), gt_ptr(a[8]), n, d, GRAPH_MAX if c.aggr == AGG_MAX else GRAPH_AGG, Args(),
    )
    # the layer output before the normalisation: pre / gpre with normalize, else out / g
    var ov = a[20] if normalize else a[3]
    var og = a[21] if normalize else a[0]
    var o: Tensor
    if c.root != 0:
        var wr = tape.variable_into(ex, gt_ptr(a[17]), gt_ptr(a[18]), f, d)
        o = tape.custom_into(ex, [agg, wl, bl, x, wr], gt_ptr(ov), gt_ptr(og), n, f, GRAPH_SAGE_OUT, Args())
    else:
        o = tape.custom_into(ex, [agg, wl, bl], gt_ptr(ov), gt_ptr(og), n, f, GRAPH_SAGE_OUT, Args())
    var y = o
    if normalize:
        y = tape.custom_into(ex, [o], gt_ptr(a[3]), gt_ptr(a[0]), n, f, GRAPH_L2, Args())
    var seed = tape.variable(ex, gt_ptr(a[0]), n, f, False)
    tape.backward_with_rules(ex, y, seed, rules)
    tape.release(ex)
    ex.sync()
