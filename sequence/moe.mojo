# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""A mixture-of-experts feed-forward block, HF transformers'
`models/mixtral/modeling_mixtral.py` (`MixtralTopKRouter`, `MixtralExperts`,
`MixtralSparseMoeBlock`): router logits x W_g^T, softmax, the top-k experts
with their probabilities renormalised to sum to one, each expert
down_proj(act(gate) * up) with (gate, up) = x gate_up_proj^T split in two,
the outputs weighted and summed. Forward only, float32.

THE SEAM IS THE ROUTING TIE: torch.topk's order among equal probabilities is
unspecified; here the k picks are made in turn by strict greater-than, so of
equal probabilities the LOWER expert index wins. The weighted sum runs over
the k picks in pick order (the reference accumulates by expert index through
index_add_, a different order of the same k terms). act is SiLU (Mixtral's
hidden_act) through the portable seam."""
from sequence.ops import FP, Args, add, fma3, ld, mul, st, sub
from checks.numerics import ftz, identical_div, identical_exp, identical_silu


@always_inline
def div(a: Float32, b: Float32) -> Float32:
    return ftz(identical_div(a, b))


def op_moe_route(t: Int, a: Args):
    """Token t: p0 x [T, D], p1 router W_g [E, D]; p2 logits out [T, E],
    p3 selected experts out [T, k] (as floats), p4 weights out [T, k];
    p5 scratch probabilities [T, E]. i0 D, i1 E, i2 k, i3 renormalise."""
    var D = a.i0
    var E = a.i1
    var k = a.i2
    var mx = Float32(0.0)
    for e in range(E):
        var s = Float32(0.0)
        for d in range(D):
            s = fma3(ld(a.p0, t * D + d), ld(a.p1, e * D + d), s)
        st(a.p2, t * E + e, s)
        if e == 0 or s > mx:
            mx = s
    var z = Float32(0.0)
    for e in range(E):
        var v = ftz(identical_exp(sub(ld(a.p2, t * E + e), mx)))
        st(a.p5, t * E + e, v)
        z = add(z, v)
    for e in range(E):
        st(a.p5, t * E + e, div(ld(a.p5, t * E + e), z))
    var tot = Float32(0.0)
    for j in range(k):
        var best = -1
        var bv = Float32(0.0)
        for e in range(E):
            var taken = False
            for i in range(j):
                if Int(a.p3.unsafe_load(t * k + i)) == e:
                    taken = True
            if taken:
                continue
            var v = ld(a.p5, t * E + e)
            if best < 0 or v > bv:
                best = e
                bv = v
        a.p3.unsafe_store(t * k + j, Float32(best))
        st(a.p4, t * k + j, bv)
        tot = add(tot, bv)
    if a.i3 != 0:
        for j in range(k):
            st(a.p4, t * k + j, div(ld(a.p4, t * k + j), tot))


def op_moe_hidden(t: Int, a: Args):
    """Element t = (token, pick, f): p0 x [T, D], p1 gate_up [E, 2F, D],
    p2 selected [T, k]; p3 h out [T, k, F] = silu(gate) * up.
    i0 D, i1 F, i2 k."""
    var D = a.i0
    var F = a.i1
    var k = a.i2
    var tok = t // (k * F)
    var rem = t - tok * k * F
    var j = rem // F
    var f = rem - j * F
    var e = Int(a.p2.unsafe_load(tok * k + j))
    var gw = a.p1 + (e * 2 * F + f) * D
    var uw = a.p1 + (e * 2 * F + F + f) * D
    var g = Float32(0.0)
    var u = Float32(0.0)
    for d in range(D):
        var x = ld(a.p0, tok * D + d)
        g = fma3(x, ld(gw, d), g)
        u = fma3(x, ld(uw, d), u)
    st(a.p3, t, mul(ftz(identical_silu(g)), u))


def op_moe_out(t: Int, a: Args):
    """Element t = (token, d): y = sum over picks j (in pick order) of
    w_j (h_j . down[e_j, d, :]). p0 h [T, k, F], p1 down [E, D, F],
    p2 selected [T, k], p3 weights [T, k], p4 y out [T, D]; i0 D, i1 F, i2 k."""
    var D = a.i0
    var F = a.i1
    var k = a.i2
    var tok = t // D
    var d = t - tok * D
    var y = Float32(0.0)
    for j in range(k):
        var e = Int(a.p2.unsafe_load(tok * k + j))
        var dw = a.p1 + (e * D + d) * F
        var hj = a.p0 + (tok * k + j) * F
        var s = Float32(0.0)
        for f in range(F):
            s = fma3(ld(hj, f), ld(dw, f), s)
        y = fma3(ld(a.p3, tok * k + j), s, y)
    st(a.p4, t, y)
