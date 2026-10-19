# SPDX-License-Identifier: Apache-2.0
# MoE block backward on the tape (lane ag-moe-backward, 2026-10-19;
# TAPE_EVERYWHERE_PLAN.md path 5). The kernels are sequence/moe_backward.mojo.
#
# NEW CAPABILITY, NEW BITS: there is no hand MoE backward, so the tape owns
# the structure and every VJP is a new kernel; the words are defined by the
# fold orders below, the same on the host column and every GPU column (the
# bodies are Exec element functions), one bit version with all columns.
# -D MOJOLEARN_AUTOGRAD_MOE_OFF restores the old refusal of the backward.
#
# ---------------------------------------------------------------- the tape graph
#   x      (variable_into dx)        [T, D]
#   router (variable_into drouter)   [E, D]
#   gate_up(variable_into dgate_up) [E, 2F * D]
#   down   (variable_into ddown)     [E, D * F]
#   w = MOE_ROUTE(x, router)          [T, k]   logits, softmax, top-k, renormalised
#                                              weights (OP_MOE_ROUTE, the forward's
#                                              own launch); the picks and logits are
#                                              saved (non-differentiable, outside the
#                                              tape: the picks are piecewise constant)
#   h = MOE_HIDDEN(x, gate_up)        [P, F]   silu(gate) * up per pair (the gather
#                                              of x by the routing order inside)
#   y = MOE_OUT(h, down, w)           [T, D]   grad storage = the caller's dy (the seed,
#                                              read in place, no copy)
# Between the route and the hidden product the driver groups the pairs by
# expert (stable: each expert's pairs in ascending pair index), once; the
# hidden and out VJPs fold each expert's weight gradient over that order.
#
# ---------------------------------------------------------------- accumulation order
# x has two consumers. Contributions are applied in decreasing consumer
# index (autograd/tape.mojo): MOE_HIDDEN's dx (the k picks' terms added in
# ascending pick order) is written into dx first, then MOE_ROUTE's dx_r goes
# to scratch and the tape's OP_AG_ACCUMULATE adds it: dx = dx_e + dx_r.
# Every other gradient has one consumer and is written once in place. Inside
# the rules: dw[p] = h[p] . Q[p] over f; ddown and dgate_up over each
# expert's pairs ascending; drouter over tokens ascending; every
# contraction the identical GEMM's contract cell (leaves + balanced tree).
#
# ---------------------------------------------------------------- launch inventory
# forward: OP_MOE_ROUTE (T), grouping COUNT (E nb) + TOTAL (E) + BASE
# (E nb + 1) + SCATTER (P), HIDDEN_FWD (P F), OUT_FWD (T D)        = 7
# backward: OUT: Q+dh (P F), DW (P), DDOWN (E D F); HIDDEN: DGU (P F), DX
# (T D), DGATEUP (E 2F D); ROUTE: ROUTE_VJP (T E), ROUTER_DX (T D),
# ROUTER_DW (E D); the tape's ACCUMULATE of dx (T D)                  = 10
# No zero fills (every rule writes every element of every destination,
# an expert with no pairs included), no host waits but the closing sync.
from std.sys.compile import is_defined

from autograd.tape import Tape, Tensor, BackwardRules
from sequence.exec_trait import Exec
from sequence.ops import FP, Args, OP_MOE_ROUTE
from sequence.moe_backward import (
    MOEB_GROUP_BLOCK,
    OP_MOEB_DDOWN,
    OP_MOEB_DGATEUP,
    OP_MOEB_DGU,
    OP_MOEB_DW,
    OP_MOEB_DX,
    OP_MOEB_GROUP_BASE,
    OP_MOEB_GROUP_COUNT,
    OP_MOEB_GROUP_SCATTER,
    OP_MOEB_GROUP_TOTAL,
    OP_MOEB_HIDDEN_FWD,
    OP_MOEB_OUT_FWD,
    OP_MOEB_Q,
    OP_MOEB_ROUTE_VJP,
    OP_MOEB_ROUTER_DW,
    OP_MOEB_ROUTER_DX,
    moeb_group_blocks,
)

# The MoE backward exists by default; MOJOLEARN_AUTOGRAD_MOE_OFF refuses it as before.
comptime AUTOGRAD_MOE = not is_defined["MOJOLEARN_AUTOGRAD_MOE_OFF"]()

# Tags 112-114 (LayerNorm 1, recurrent 16-18, transformer 32-47, Mamba 48-63, LM 64-71).
comptime MOE_ROUTE = 112
comptime MOE_HIDDEN = 113
comptime MOE_OUT = 114

#: the pair indices travel as float words: exact below 2^24
comptime MOE_MAX_PAIRS = 1 << 24


@fieldwise_init
struct MoeRules(BackwardRules):
    """The MoE block's VJPs. Holds the routing (picks, logits) and the
    grouping (order, poff) the forward left on the executor; they must
    outlive the backward."""

    var T: Int
    var D: Int
    var F: Int
    var En: Int
    var k: Int
    var renorm: Int
    var sel: FP
    var logits: FP
    var order: FP
    var poff: FP

    def supports(self, tag: Int) -> Bool:
        return tag >= MOE_ROUTE and tag <= MOE_OUT

    def zeroed_destination(self, tag: Int, saved: Args, edge: Int) -> Bool:
        # every launch below stores every element of the destination it writes
        return False

    def writes_total(self, tag: Int, saved: Args, edge: Int) -> Bool:
        # the plain contract: the tape adds x's two contributions
        return False

    def apply[E: Exec](
        self, mut ex: E, tag: Int, saved: Args, output: FP, upstream: FP,
        inputs: List[FP], gradients: List[FP], rows: List[Int], cols: List[Int],
    ) raises:
        if tag == MOE_OUT:
            self._out(ex, upstream, inputs, gradients)
        elif tag == MOE_HIDDEN:
            self._hidden(ex, upstream, inputs, gradients)
        elif tag == MOE_ROUTE:
            self._route(ex, output, upstream, inputs, gradients)
        else:
            raise Error("unknown MoE backward rule")

    def _out[E: Exec](self, mut ex: E, dy: FP, inputs: List[FP], gradients: List[FP]) raises:
        # inputs [h, down, w]; gradients [dh, ddown, dw]
        var P = self.T * self.k
        var q = ex.alloc(P * self.F)
        var a = Args()
        a.p0 = dy
        a.p1 = inputs[1]
        a.p2 = self.sel
        a.p3 = q
        a.p4 = inputs[2]
        a.p5 = gradients[0]
        a.i0 = self.D
        a.i1 = self.F
        a.i2 = self.k
        ex.launch[OP_MOEB_Q](a, P * self.F)
        var b = Args()
        b.p0 = inputs[0]
        b.p1 = q
        b.p2 = gradients[2]
        b.i1 = self.F
        ex.launch[OP_MOEB_DW](b, P)
        var c = Args()
        c.p0 = dy
        c.p1 = inputs[0]
        c.p2 = inputs[2]
        c.p3 = self.order
        c.p4 = self.poff
        c.p5 = gradients[1]
        c.i0 = self.D
        c.i1 = self.F
        c.i2 = self.k
        ex.launch[OP_MOEB_DDOWN](c, self.En * self.D * self.F)
        ex.free(q)

    def _hidden[E: Exec](self, mut ex: E, dh: FP, inputs: List[FP], gradients: List[FP]) raises:
        # inputs [x, gate_up]; gradients [dx, dgate_up]
        var P = self.T * self.k
        var dgu = ex.alloc(P * 2 * self.F)
        var a = Args()
        a.p0 = inputs[0]
        a.p1 = inputs[1]
        a.p2 = self.sel
        a.p4 = dh
        a.p5 = dgu
        a.i0 = self.D
        a.i1 = self.F
        a.i2 = self.k
        ex.launch[OP_MOEB_DGU](a, P * self.F)
        var b = Args()
        b.p0 = dgu
        b.p1 = inputs[1]
        b.p2 = self.sel
        b.p3 = gradients[0]
        b.i0 = self.D
        b.i1 = self.F
        b.i2 = self.k
        ex.launch[OP_MOEB_DX](b, self.T * self.D)
        var c = Args()
        c.p0 = dgu
        c.p1 = inputs[0]
        c.p3 = self.order
        c.p4 = self.poff
        c.p5 = gradients[1]
        c.i0 = self.D
        c.i1 = self.F
        c.i2 = self.k
        ex.launch[OP_MOEB_DGATEUP](c, self.En * 2 * self.F * self.D)
        ex.free(dgu)

    def _route[E: Exec](self, mut ex: E, w: FP, dw: FP, inputs: List[FP], gradients: List[FP]) raises:
        # inputs [x, router]; gradients [dx, drouter]; output w
        var dl = ex.alloc(self.T * self.En)
        var a = Args()
        a.p0 = self.logits
        a.p1 = self.sel
        a.p2 = w
        a.p3 = dw
        a.p4 = dl
        a.i1 = self.En
        a.i2 = self.k
        a.i3 = self.renorm
        ex.launch[OP_MOEB_ROUTE_VJP](a, self.T * self.En)
        var b = Args()
        b.p0 = dl
        b.p1 = inputs[1]
        b.p2 = gradients[0]
        b.i0 = self.D
        b.i1 = self.En
        ex.launch[OP_MOEB_ROUTER_DX](b, self.T * self.D)
        var c = Args()
        c.p0 = dl
        c.p1 = inputs[0]
        c.p2 = gradients[1]
        c.i0 = self.D
        c.i1 = self.En
        c.i2 = self.T
        ex.launch[OP_MOEB_ROUTER_DW](c, self.En * self.D)
        ex.free(dl)


def moe_group_pairs[E: Exec](mut ex: E, sel: FP, order: FP, poff: FP, n_pairs: Int, n_experts: Int) raises:
    """The pairs grouped by expert, stable: order[poff[e] .. poff[e + 1])
    holds expert e's pair indices ascending (as float words). Integer
    counts only, so the result is exact on every executor."""
    var nb = moeb_group_blocks(n_pairs)
    var cnt = ex.alloc(n_experts * nb)
    var tot = ex.alloc(n_experts)
    var base = ex.alloc(n_experts * nb)
    var a = Args()
    a.p0 = sel
    a.p1 = cnt
    a.p2 = tot
    a.p3 = poff
    a.p4 = base
    a.p5 = order
    a.i0 = n_pairs
    a.i1 = n_experts
    a.i2 = nb
    ex.launch[OP_MOEB_GROUP_COUNT](a, n_experts * nb)
    ex.launch[OP_MOEB_GROUP_TOTAL](a, n_experts)
    ex.launch[OP_MOEB_GROUP_BASE](a, n_experts * nb + 1)
    ex.launch[OP_MOEB_GROUP_SCATTER](a, n_pairs)
    ex.free(base)
    ex.free(tot)
    ex.free(cnt)


def moe_tape_train_step[E: Exec](
    mut ex: E, X: FP, Wg: FP, Gu: FP, Dn: FP, DY: FP,
    DX: FP, DWg: FP, DGu: FP, DDn: FP, Y: FP, L: FP, Sel: FP, W: FP,
    T: Int, D: Int, F: Int, En: Int, k: Int, renorm: Int,
) raises:
    """One MoE block forward and backward on the tape, every buffer on
    `ex`'s side: from x [T, D], the router [E, D], gate_up [E, 2F, D],
    down [E, D, F] and the output cotangent dy [T, D], the forward's y,
    logits, picks and weights into Y, L, Sel, W and the gradients of x and
    the three weights into DX, DWg, DGu, DDn. Ends on a sync."""
    comptime if not AUTOGRAD_MOE:
        raise Error("moe_backward: the backward pass is not available in this build")
    if T < 1 or D < 1 or F < 1 or En < 1 or k < 1 or k > En:
        raise Error("moe_backward: T, D, F, E >= 1 and 1 <= k <= E")
    var P = T * k
    if P >= MOE_MAX_PAIRS:
        raise Error("moe_backward: T * k must be below 2^24")
    # ---- forward: the route (the forward's own launch), the grouping, the products
    var Pr = ex.alloc(T * En)
    var a = Args()
    a.p0 = X
    a.p1 = Wg
    a.p2 = L
    a.p3 = Sel
    a.p4 = W
    a.p5 = Pr
    a.i0 = D
    a.i1 = En
    a.i2 = k
    a.i3 = renorm
    ex.launch[OP_MOE_ROUTE](a, T)
    var order = ex.alloc(P)
    var poff = ex.alloc(En + 1)
    moe_group_pairs(ex, Sel, order, poff, P, En)
    var H = ex.alloc(P * F)
    var b = Args()
    b.p0 = X
    b.p1 = Gu
    b.p2 = Sel
    b.p3 = H
    b.i0 = D
    b.i1 = F
    b.i2 = k
    ex.launch[OP_MOEB_HIDDEN_FWD](b, P * F)
    var c = Args()
    c.p0 = H
    c.p1 = Dn
    c.p2 = Sel
    c.p3 = W
    c.p4 = Y
    c.i0 = D
    c.i1 = F
    c.i2 = k
    ex.launch[OP_MOEB_OUT_FWD](c, T * D)
    # ---- record (no launches) and differentiate
    var tape = Tape()
    var x = tape.variable_into(ex, X, DX, T, D)
    var wg = tape.variable_into(ex, Wg, DWg, En, D)
    var gu = tape.variable_into(ex, Gu, DGu, En, 2 * F * D)
    var dn = tape.variable_into(ex, Dn, DDn, En, D * F)
    var w = tape.custom(ex, [x, wg], W, T, k, MOE_ROUTE, Args())
    var h = tape.custom(ex, [x, gu], H, P, F, MOE_HIDDEN, Args())
    var y = tape.custom_into(ex, [h, dn, w], Y, DY, T, D, MOE_OUT, Args())
    var seed = tape.variable(ex, DY, T, D, False)
    var rules = MoeRules(T, D, F, En, k, renorm, Sel, L, order, poff)
    tape.backward_with_rules(ex, y, seed, rules)
    tape.release(ex)
    ex.sync()
    ex.free(H)
    ex.free(poff)
    ex.free(order)
    ex.free(Pr)
