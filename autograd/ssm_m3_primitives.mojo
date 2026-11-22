# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Mamba-3 trapezoid core as a generic differentiable tape operation (lane neural-blocks-3-m3,
2026-11-22). Recorded as a custom node with tag GRAPH_TRAPEZOID_SCAN, answered by
`autograd/ssm_primitives.mojo::SsmRules`; element bodies `autograd/kernels/mamba3_trapezoid_body.mojo`
through `autograd/ssm_m3_ops.mojo` on any executor (host column and device kernel, the same functions).

graph_trapezoid_scan  the Mamba-3 SISO state-space recurrence with the trapezoid rule (the decode
                      step's arithmetic, mamba/impl/modules/mamba3.mojo m3_step_core_kernel), from a
                      zero state:
                        alpha = exp(A dt), sigma = sigmoid(trap), gamma = dt sigma,
                        beta = dt (1 - sigma) alpha,
                        h[t] = alpha h[t-1] + beta k[t-1] v[t-1] + gamma k[t] v[t],
                        y[t] = q[t] . h[t] + D v[t]
                      per head; v (.., heads * P), k and q (.., heads * N), dt, A, trap (.., heads).
"""
from sequence.exec_trait import Exec
from sequence.ops import Args, FP
from autograd.tape import Tape, Tensor
from autograd.ssm_m3_ops import (
    OP_SSM_M3_SCAN,
    OP_SSM_M3_BWD,
    OP_SSM_M3_DQ,
    OP_SSM_M3_DK,
    OP_SSM_M3_COEF,
    OP_SSM_M3_DV,
    OP_SSM_M3_DD,
)

comptime GRAPH_TRAPEZOID_SCAN = 242


def graph_trapezoid_scan[E: Exec](
    mut tape: Tape, mut ex: E, v: Tensor, k: Tensor, q: Tensor, dt: Tensor, a: Tensor, tr: Tensor,
    d_skip: Tensor, heads: Int,
) raises -> Tensor:
    """The Mamba-3 trapezoid scan over (batch, length) sequences.

    v: (..., length, heads * P); k, q: (..., length, heads * N); dt, a, tr: (..., length, heads);
    d_skip: heads values. Returns y (..., length, heads * P).

    Serial by nature over time; each (sequence, head, channel) chain is one element (one GPU thread),
    the decode step's shape over a whole sequence."""
    var shape = tape.shape_of(v)
    if shape.ndim < 2:
        raise Error("ValueError: trapezoid_scan: v needs (length, features) axes")
    if heads < 1:
        raise Error("ValueError: trapezoid_scan: heads must be positive")
    var m = tape.rows(v)
    var length = shape.d[shape.ndim - 2]
    if length < 1 or m % length != 0:
        raise Error("ValueError: trapezoid_scan: the rows of v must be whole sequences")
    var batch = m // length
    var vw = tape.cols(v)
    var kw = tape.cols(k)
    if vw % heads != 0 or kw % heads != 0 or vw < heads or kw < heads:
        raise Error("ValueError: trapezoid_scan: v and k must hold whole heads")
    var pd = vw // heads
    var ns = kw // heads
    if tape.rows(k) != m or tape.rows(q) != m or tape.cols(q) != kw:
        raise Error("ValueError: trapezoid_scan: k and q must be (rows of v, heads * state)")
    if (tape.rows(dt) != m or tape.cols(dt) != heads or tape.rows(a) != m or tape.cols(a) != heads
            or tape.rows(tr) != m or tape.cols(tr) != heads):
        raise Error("ValueError: trapezoid_scan: dt, A and trap must be (rows of v, heads)")
    if tape.rows(d_skip) * tape.cols(d_skip) != heads:
        raise Error("ValueError: trapezoid_scan: D must hold one value per head")
    var vp = tape.value_of(ex, v)
    var kp = tape.value_of(ex, k)
    var qp = tape.value_of(ex, q)
    var dtp = tape.value_of(ex, dt)
    var ap = tape.value_of(ex, a)
    var trp = tape.value_of(ex, tr)
    var skp = tape.value_of(ex, d_skip)
    var hs = tape.reserve(ex, m * heads * pd * ns)  # the state after every token
    var output = tape.reserve(ex, m * heads * pd)
    var s = Args()
    s.p0 = output
    s.p1 = hs
    s.p2 = vp
    s.p3 = kp
    s.p4 = qp
    s.p5 = dtp
    s.p6 = ap
    s.p7 = trp
    s.p8 = skp
    s.i0 = batch
    s.i1 = length
    s.i2 = heads
    s.i3 = pd
    s.i4 = ns
    ex.launch[OP_SSM_M3_SCAN](s, batch * heads * pd)
    var saved = Args()
    saved.i0 = batch
    saved.i1 = length
    saved.i2 = heads
    saved.i3 = pd
    saved.i4 = ns
    saved.p0 = hs
    if tape.is_recording():
        saved.p1 = tape.reserve(ex, m * heads * pd * ns)  # g, the state's gradient
        saved.p2 = tape.reserve(ex, m * heads * pd)  # v's scan gradient
        saved.p3 = tape.reserve(ex, 3 * m * heads * pd)  # d alpha, d beta, d gamma per channel
    else:
        saved.p1 = hs
        saved.p2 = hs
        saved.p3 = hs
    return tape.custom_nd(ex, [v, k, q, dt, a, tr, d_skip], output, shape, GRAPH_TRAPEZOID_SCAN, saved)


def trapezoid_scan_bwd[E: Exec](mut ex: E, saved: Args, up: FP, inputs: List[FP], gradients: List[FP]) raises:
    """The backward of graph_trapezoid_scan: every destination written whole."""
    var batch = saved.i0
    var length = saved.i1
    var nh = saved.i2
    var pd = saved.i3
    var ns = saved.i4
    var m = batch * length
    var b = Args()
    b.p0 = saved.p1
    b.p1 = saved.p2
    b.p2 = saved.p3
    b.p3 = up
    b.p4 = saved.p0
    b.p5 = inputs[0]
    b.p6 = inputs[1]
    b.p7 = inputs[2]
    b.p8 = inputs[3]
    b.p9 = inputs[4]
    b.p10 = inputs[5]
    b.i0 = batch
    b.i1 = length
    b.i2 = nh
    b.i3 = pd
    b.i4 = ns
    ex.launch[OP_SSM_M3_BWD](b, batch * nh * pd)
    var dq = Args()
    dq.p0 = gradients[2]
    dq.p1 = up
    dq.p2 = saved.p0
    dq.i0 = m
    dq.i2 = nh
    dq.i3 = pd
    dq.i4 = ns
    ex.launch[OP_SSM_M3_DQ](dq, m * nh * ns)
    var dk = Args()
    dk.p0 = gradients[1]
    dk.p1 = saved.p1
    dk.p2 = inputs[0]
    dk.p3 = inputs[3]
    dk.p4 = inputs[4]
    dk.p5 = inputs[5]
    dk.i0 = m
    dk.i1 = length
    dk.i2 = nh
    dk.i3 = pd
    dk.i4 = ns
    ex.launch[OP_SSM_M3_DK](dk, m * nh * ns)
    var c = Args()
    c.p0 = gradients[3]
    c.p1 = gradients[4]
    c.p2 = gradients[5]
    c.p3 = saved.p3
    c.p4 = inputs[3]
    c.p5 = inputs[4]
    c.p6 = inputs[5]
    c.i0 = m
    c.i2 = nh
    c.i3 = pd
    ex.launch[OP_SSM_M3_COEF](c, m * nh)
    var dv = Args()
    dv.p0 = gradients[0]
    dv.p1 = saved.p2
    dv.p2 = up
    dv.p3 = inputs[6]
    dv.i0 = m
    dv.i2 = nh
    dv.i3 = pd
    ex.launch[OP_SSM_M3_DV](dv, m * nh * pd)
    var dd = Args()
    dd.p0 = gradients[6]
    dd.p1 = up
    dd.p2 = inputs[0]
    dd.i0 = m
    dd.i2 = nh
    dd.i3 = pd
    ex.launch[OP_SSM_M3_DD](dd, nh)
