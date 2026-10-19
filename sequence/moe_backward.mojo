# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The MoE block's training forward and its vector-Jacobian products
(lane ag-moe-backward, 2026-10-19; TAPE_EVERYWHERE_PLAN.md path 5). HOST-SAFE:
every body below is an element function `(t, Args)` that HostExec and
DeviceExec both run through `sequence/dispatch.mojo` (codes 2900..2914),
so the host column and every GPU column compute the same words by
construction. The tape assembly (graph, rules, driver) is
`autograd/route_ops.mojo`.

Notation: T tokens, D model width, F expert width, E experts, k picks;
a pair p = tok * k + j (token tok's j-th pick), P = T * k pairs, e_p its
expert (`sel[p]`, a float holding an integer), w_p its routing weight.

FORWARD (training): the forward's own route (OP_MOE_ROUTE: logits,
softmax, top-k, renormalised weights), then the pairs grouped by expert
(OP_MOEB_GROUP_*: a stable counting sort, integer counts, exact), then
h[p, f] = silu(g) * u (OP_MOEB_HIDDEN_FWD = `op_moe_hidden`, pair order)
and y[tok, d] = sum_j w_j (h_j . down[e_j, d, :]) (OP_MOEB_OUT_FWD =
`op_moe_out`). Those two are the host column's forward bodies; under the
IDENTICAL default (MOE_GROUPED_GEMM) their cells are the identical GEMM's
contract cell, the words the device's grouped forward stores, so the
training forward's y equals MoEBlock.forward's.

BACKWARD (NEW BITS, a capability with no earlier version; every column
together). Every contraction is the identical GEMM's contract cell
(`moe_bwd_cell`: leaves of contract_leaf_size(n) ascending terms, each an
`identical_mul_add` chain from +0.0, the leaves on the contract's balanced
tree, `_fold_push16` / `_fold_drain16`), one output cell a thread:
  Q[p, f]     = dy[tok, :] . down[e_p, :, f]           over d   (OP_MOEB_Q)
  dh[p, f]    = w_p * Q[p, f]                                    (same launch)
  dw[p]       = h[p, :] . Q[p, :]                      over f   (OP_MOEB_DW)
  dgu[p, f]   = dh * u * silu'(g),  dgu[p, F + f] = dh * silu(g) (OP_MOEB_DGU;
                g, u recomputed with the forward's cell)
  dx_e[tok,d] = sum_j (dgu[p_j, :] . gate_up[e_j, :, d])  over r, then the
                k picks added in ascending pick order: the gather's
                scatter-add VJP in a fixed order               (OP_MOEB_DX)
  ddown[e,d,f]= sum over e's pairs, ascending pair index, of
                (w_p * dy[tok, d]) * h[p, f]           (OP_MOEB_DDOWN)
  dgate_up[e,r,d] = sum over e's pairs, ascending, of dgu[p, r] * x[tok, d]
                                                         (OP_MOEB_DGATEUP)
  route: dL[t, e] from dw through the renormalisation and the softmax,
  the top-k selection passing gradient to the selected probabilities only
  (OP_MOEB_ROUTE_VJP; the probabilities recomputed from the stored logits
  with op_moe_route's words); dx_r = dL . router over e (OP_MOEB_ROUTER_DX);
  drouter[e, d] = sum over tokens ascending of dL[t, e] x[t, d]
  (OP_MOEB_ROUTER_DW).
An expert with no pairs gets +0.0 gradients (an empty cell). The x
gradient's two terms (experts, router) are added by the tape
(autograd/route_ops.mojo, accumulation order)."""
from checks.numerics import ftz, identical_div, identical_exp, identical_mul_add, identical_sigmoid, identical_silu
from gemm.contract import contract_leaf_size, leaf_count
from sequence.ops import FP, Args, add, fma3, ld, mul, st, sub
from sequence.moe import op_moe_hidden, op_moe_out
from sequence.moe_grouped_fold import MOE_FOLD_SLOTS, MOE_GROUPED_GEMM, _fold_drain16, _fold_push16, moe_contract_cell

comptime OP_MOEB_HIDDEN_FWD = 2900
comptime OP_MOEB_OUT_FWD = 2901
comptime OP_MOEB_GROUP_COUNT = 2902
comptime OP_MOEB_GROUP_TOTAL = 2903
comptime OP_MOEB_GROUP_BASE = 2904
comptime OP_MOEB_GROUP_SCATTER = 2905
comptime OP_MOEB_Q = 2906
comptime OP_MOEB_DW = 2907
comptime OP_MOEB_DGU = 2908
comptime OP_MOEB_DX = 2909
comptime OP_MOEB_DDOWN = 2910
comptime OP_MOEB_DGATEUP = 2911
comptime OP_MOEB_ROUTE_VJP = 2912
comptime OP_MOEB_ROUTER_DX = 2913
comptime OP_MOEB_ROUTER_DW = 2914
comptime OP_MOEB_FIRST = OP_MOEB_HIDDEN_FWD
comptime OP_MOEB_LAST = OP_MOEB_ROUTER_DW

#: pairs per counting block of the grouping (a constant: no shape rule)
comptime MOEB_GROUP_BLOCK = 64


def moeb_group_blocks(n_pairs: Int) -> Int:
    """The grouping's block count, ceil(P / MOEB_GROUP_BLOCK)."""
    return max((n_pairs + MOEB_GROUP_BLOCK - 1) // MOEB_GROUP_BLOCK, 1)


@always_inline
def _mdiv(a: Float32, b: Float32) -> Float32:
    return ftz(identical_div(a, b))


@always_inline
def moe_bwd_cell(a: FP, sa: Int, b: FP, sb: Int, n: Int) -> Float32:
    """sum over i < n of a[i * sa] * b[i * sb] as the identical GEMM's
    contract cell (`moe_contract_cell` with strides; sa = sb = 1 is that
    cell word for word)."""
    var leaf = contract_leaf_size(n)
    var p_count = leaf_count(n, leaf)
    var stack = InlineArray[Float32, MOE_FOLD_SLOTS](fill=Float32(0.0))
    var occ = 0
    for t in range(p_count):
        var pb = t * leaf
        var pe = pb + leaf
        if pe > n:
            pe = n
        var acc = Float32(0.0)
        for i in range(pb, pe):
            acc = ftz(identical_mul_add(ftz(a.unsafe_load(i * sa)), ftz(b.unsafe_load(i * sb)), acc))
        _fold_push16(stack, occ, ftz(acc))
    return ftz(_fold_drain16(stack, occ))


@always_inline
def moe_bwd_segment_cell[SCALE: Bool](
    order: FP, s0: Int, n: Int, k: Int,
    a: FP, a_pair: Int, a_tok: Int, w: FP,
    b: FP, b_pair: Int, b_tok: Int,
) -> Float32:
    """The contract cell over one expert's pairs, order[s0 .. s0 + n)
    (ascending pair index): term i = A(p) * B(p) with A(p) = a[p * a_pair +
    tok * a_tok] (times w[p] first when SCALE) and B(p) = b[p * b_pair +
    tok * b_tok], tok = p // k."""
    var leaf = contract_leaf_size(n)
    var p_count = leaf_count(n, leaf)
    var stack = InlineArray[Float32, MOE_FOLD_SLOTS](fill=Float32(0.0))
    var occ = 0
    for t in range(p_count):
        var pb = t * leaf
        var pe = pb + leaf
        if pe > n:
            pe = n
        var acc = Float32(0.0)
        for i in range(pb, pe):
            var p = Int(order.unsafe_load(s0 + i))
            var tok = p // k
            var av = ftz(a.unsafe_load(p * a_pair + tok * a_tok))
            comptime if SCALE:
                av = mul(ld(w, p), av)
            var bv = ftz(b.unsafe_load(p * b_pair + tok * b_tok))
            acc = ftz(identical_mul_add(av, bv, acc))
        _fold_push16(stack, occ, ftz(acc))
    return ftz(_fold_drain16(stack, occ))


@always_inline
def _gate_up(xr: FP, gw: FP, uw: FP, D: Int) -> Tuple[Float32, Float32]:
    """op_moe_hidden's g and u, the same words (both of its spellings)."""
    var g = Float32(0.0)
    var u = Float32(0.0)
    comptime if MOE_GROUPED_GEMM:
        g = moe_contract_cell(xr, gw, D)
        u = moe_contract_cell(xr, uw, D)
    else:
        for d in range(D):
            var x = ld(xr, d)
            g = fma3(x, ld(gw, d), g)
            u = fma3(x, ld(uw, d), u)
    return (g, u)


# ------------------------------------------------------------ grouping
@always_inline
def op_moeb_group_count(i: Int, a: Args):
    """Element (e, b), i = e * nb + b: p1[i] = the pairs of block b whose
    expert is e. p0 sel [P]; i0 P, i1 E, i2 nb."""
    var nb = a.i2
    var e = i // nb
    var b = i - e * nb
    var lo = b * MOEB_GROUP_BLOCK
    var hi = min(lo + MOEB_GROUP_BLOCK, a.i0)
    var c = 0
    for q in range(lo, hi):
        if Int(a.p0.unsafe_load(q)) == e:
            c += 1
    a.p1.unsafe_store(i, Float32(c))


@always_inline
def op_moeb_group_total(e: Int, a: Args):
    """Element e: p2[e] = expert e's pairs. p1 counts [E, nb]; i2 nb."""
    var nb = a.i2
    var c = 0
    for b in range(nb):
        c += Int(a.p1.unsafe_load(e * nb + b))
    a.p2.unsafe_store(e, Float32(c))


@always_inline
def op_moeb_group_base(i: Int, a: Args):
    """Element i < E * nb: p4[i] = the first slot of (expert e, block b)'s
    pairs; b = 0 also writes poff p3[e]. Element E * nb writes p3[E] = P.
    p1 counts, p2 totals; i0 P, i1 E, i2 nb."""
    var nb = a.i2
    if i == a.i1 * nb:
        a.p3.unsafe_store(a.i1, Float32(a.i0))
        return
    var e = i // nb
    var b = i - e * nb
    var s = 0
    for c in range(e):
        s += Int(a.p2.unsafe_load(c))
    for c in range(b):
        s += Int(a.p1.unsafe_load(e * nb + c))
    a.p4.unsafe_store(i, Float32(s))
    if b == 0:
        a.p3.unsafe_store(e, Float32(s))


@always_inline
def op_moeb_group_scatter(p: Int, a: Args):
    """Element p: order p5[slot] = p, slot = its (expert, block) base plus
    the earlier pairs of its block with its expert: a stable sort, each
    expert's pairs in ascending pair index. p0 sel, p4 bases; i2 nb."""
    var nb = a.i2
    var e = Int(a.p0.unsafe_load(p))
    var b = p // MOEB_GROUP_BLOCK
    var r = Int(a.p4.unsafe_load(e * nb + b))
    for q in range(b * MOEB_GROUP_BLOCK, p):
        if Int(a.p0.unsafe_load(q)) == e:
            r += 1
    a.p5.unsafe_store(r, Float32(p))


# ------------------------------------------------------------ expert VJPs
@always_inline
def op_moeb_q(t: Int, a: Args):
    """Element (p, f): p3 Q[p, f] = dy[tok, :] . down[e_p, :, f] and
    p5 dh[p, f] = w_p * Q[p, f]. p0 dy [T, D], p1 down [E, D, F],
    p2 sel [P], p4 w [P]; i0 D, i1 F, i2 k."""
    var D = a.i0
    var F = a.i1
    var p = t // F
    var f = t - p * F
    var tok = p // a.i2
    var e = Int(a.p2.unsafe_load(p))
    var q = moe_bwd_cell(a.p0 + tok * D, 1, a.p1 + e * D * F + f, F, D)
    st(a.p3, t, q)
    st(a.p5, t, mul(ld(a.p4, p), q))


@always_inline
def op_moeb_dw(p: Int, a: Args):
    """Element p: p2 dw[p] = h[p, :] . Q[p, :]. p0 h [P, F], p1 Q; i1 F."""
    var F = a.i1
    st(a.p2, p, moe_bwd_cell(a.p0 + p * F, 1, a.p1 + p * F, 1, F))


@always_inline
def op_moeb_dgu(t: Int, a: Args):
    """Element (p, f): p5 dgu[p, f] = dh * u * silu'(g) and dgu[p, F + f]
    = dh * silu(g), g and u the forward's words. p0 x [T, D], p1 gate_up
    [E, 2F, D], p2 sel, p4 dh [P, F]; i0 D, i1 F, i2 k."""
    var D = a.i0
    var F = a.i1
    var p = t // F
    var f = t - p * F
    var tok = p // a.i2
    var e = Int(a.p2.unsafe_load(p))
    var gu = _gate_up(a.p0 + tok * D, a.p1 + (e * 2 * F + f) * D, a.p1 + (e * 2 * F + F + f) * D, D)
    var g = gu[0]
    var u = gu[1]
    var dh = ld(a.p4, t)
    var s = ftz(identical_silu(g))
    var sg = ftz(identical_sigmoid(g))
    # silu'(g) = sg * (1 + g * (1 - sg))
    var ds = mul(sg, add(Float32(1.0), mul(g, sub(Float32(1.0), sg))))
    st(a.p5, p * 2 * F + f, mul(mul(dh, u), ds))
    st(a.p5, p * 2 * F + F + f, mul(dh, s))


@always_inline
def op_moeb_dx(t: Int, a: Args):
    """Element (tok, d): p3 dx[tok, d] = the k picks' dgu[p_j, :] .
    gate_up[e_j, :, d], added in ascending pick order from +0.0.
    p0 dgu [P, 2F], p1 gate_up, p2 sel; i0 D, i1 F, i2 k."""
    var D = a.i0
    var F = a.i1
    var k = a.i2
    var tok = t // D
    var d = t - tok * D
    var acc = Float32(0.0)
    for j in range(k):
        var p = tok * k + j
        var e = Int(a.p2.unsafe_load(p))
        acc = add(acc, moe_bwd_cell(a.p0 + p * 2 * F, 1, a.p1 + e * 2 * F * D + d, D, 2 * F))
    st(a.p3, t, acc)


@always_inline
def op_moeb_ddown(t: Int, a: Args):
    """Element (e, d, f): p5 ddown[e, d, f] = over e's pairs ascending,
    (w_p * dy[tok, d]) * h[p, f]. p0 dy, p1 h, p2 w, p3 order, p4 poff
    [E + 1]; i0 D, i1 F, i2 k."""
    var D = a.i0
    var F = a.i1
    var e = t // (D * F)
    var rem = t - e * D * F
    var d = rem // F
    var f = rem - d * F
    var s0 = Int(a.p4.unsafe_load(e))
    var n = Int(a.p4.unsafe_load(e + 1)) - s0
    st(a.p5, t, moe_bwd_segment_cell[True](a.p3, s0, n, a.i2, a.p0 + d, 0, D, a.p2, a.p1 + f, F, 0))


@always_inline
def op_moeb_dgateup(t: Int, a: Args):
    """Element (e, r, d): p5 dgate_up[e, r, d] = over e's pairs ascending,
    dgu[p, r] * x[tok, d]. p0 dgu, p1 x, p3 order, p4 poff; i0 D, i1 F, i2 k."""
    var D = a.i0
    var R = 2 * a.i1
    var e = t // (R * D)
    var rem = t - e * R * D
    var r = rem // D
    var d = rem - r * D
    var s0 = Int(a.p4.unsafe_load(e))
    var n = Int(a.p4.unsafe_load(e + 1)) - s0
    st(a.p5, t, moe_bwd_segment_cell[False](a.p3, s0, n, a.i2, a.p0 + r, R, 0, a.p0, a.p1 + d, 0, D))


# ------------------------------------------------------------ route VJPs
@always_inline
def _row_max_sum(lr: FP, E: Int) -> Tuple[Float32, Float32]:
    """op_moe_route's max and softmax denominator of one row of logits."""
    var mx = Float32(0.0)
    for c in range(E):
        var s = ld(lr, c)
        if c == 0 or s > mx:
            mx = s
    var z = Float32(0.0)
    for c in range(E):
        z = add(z, ftz(identical_exp(sub(ld(lr, c), mx))))
    return (mx, z)


@always_inline
def _prob(lr: FP, c: Int, mx: Float32, z: Float32) -> Float32:
    """op_moe_route's probability word of expert c."""
    return _mdiv(ftz(identical_exp(sub(ld(lr, c), mx))), z)


@always_inline
def op_moeb_route_vjp(i: Int, a: Args):
    """Element (t, e): p4 dL[t, e]. With pv_j = P[sel_j], tot = their sum
    (pick order), w_j = pv_j / tot: dpv_j = (dw_j - sum_i dw_i w_i) / tot
    (renormalised; dw_j otherwise), dP = dpv at the picks and 0 elsewhere
    (top-k is piecewise constant), dL = P (dP - sum_j dpv_j pv_j).
    p0 logits [T, E], p1 sel [T, k], p2 w [T, k], p3 dw [T, k];
    i1 E, i2 k, i3 renormalise."""
    var E = a.i1
    var k = a.i2
    var t = i // E
    var e = i - t * E
    var lr = a.p0 + t * E
    var mz = _row_max_sum(lr, E)
    var mx = mz[0]
    var z = mz[1]
    var tot = Float32(0.0)
    var c = Float32(0.0)
    for j in range(k):
        tot = add(tot, _prob(lr, Int(a.p1.unsafe_load(t * k + j)), mx, z))
        c = fma3(ld(a.p3, t * k + j), ld(a.p2, t * k + j), c)
    var dot = Float32(0.0)
    var dpe = Float32(0.0)
    for j in range(k):
        var sj = Int(a.p1.unsafe_load(t * k + j))
        var dpv = ld(a.p3, t * k + j)
        if a.i3 != 0:
            dpv = _mdiv(sub(dpv, c), tot)
        dot = fma3(dpv, _prob(lr, sj, mx, z), dot)
        if sj == e:
            dpe = dpv
    st(a.p4, i, mul(_prob(lr, e, mx, z), sub(dpe, dot)))


@always_inline
def op_moeb_router_dx(t: Int, a: Args):
    """Element (tok, d): p2 dx_r[tok, d] = dL[tok, :] . router[:, d].
    p0 dL [T, E], p1 router [E, D]; i0 D, i1 E."""
    var D = a.i0
    var tok = t // D
    var d = t - tok * D
    st(a.p2, t, moe_bwd_cell(a.p0 + tok * a.i1, 1, a.p1 + d, D, a.i1))


@always_inline
def op_moeb_router_dw(t: Int, a: Args):
    """Element (e, d): p2 drouter[e, d] = over tokens ascending, dL[t, e]
    x[t, d]. p0 dL, p1 x; i0 D, i1 E, i2 T."""
    var D = a.i0
    var e = t // D
    var d = t - e * D
    st(a.p2, t, moe_bwd_cell(a.p0 + e, a.i1, a.p1 + d, D, a.i2))


@always_inline
def apply_moe_backward[OP: Int](t: Int, a: Args):
    """The OP_MOEB_FIRST..OP_MOEB_LAST family (sequence/dispatch.mojo)."""
    comptime if OP == OP_MOEB_HIDDEN_FWD:
        op_moe_hidden(t, a)
    elif OP == OP_MOEB_OUT_FWD:
        op_moe_out(t, a)
    elif OP == OP_MOEB_GROUP_COUNT:
        op_moeb_group_count(t, a)
    elif OP == OP_MOEB_GROUP_TOTAL:
        op_moeb_group_total(t, a)
    elif OP == OP_MOEB_GROUP_BASE:
        op_moeb_group_base(t, a)
    elif OP == OP_MOEB_GROUP_SCATTER:
        op_moeb_group_scatter(t, a)
    elif OP == OP_MOEB_Q:
        op_moeb_q(t, a)
    elif OP == OP_MOEB_DW:
        op_moeb_dw(t, a)
    elif OP == OP_MOEB_DGU:
        op_moeb_dgu(t, a)
    elif OP == OP_MOEB_DX:
        op_moeb_dx(t, a)
    elif OP == OP_MOEB_DDOWN:
        op_moeb_ddown(t, a)
    elif OP == OP_MOEB_DGATEUP:
        op_moeb_dgateup(t, a)
    elif OP == OP_MOEB_ROUTE_VJP:
        op_moeb_route_vjp(t, a)
    elif OP == OP_MOEB_ROUTER_DX:
        op_moeb_router_dx(t, a)
    elif OP == OP_MOEB_ROUTER_DW:
        op_moeb_router_dw(t, a)
