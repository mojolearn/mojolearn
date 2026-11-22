# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The state-space cores as generic differentiable tape operations (lane neural-blocks-2, 2026-11-22).

Andrew, 2026-11-22: every neural architecture's pieces are public building blocks, Mamba included; the
math of each operation is written once. These are typed operations over tape tensors (no block buffers),
forward and first-order backward, on any executor (host column and device kernel through
`autograd/ssm_ops.mojo`, the same element bodies), recorded as custom nodes answered by `SsmRules`.

graph_selective_scan  the Mamba-1 selective scan (`mamba_ssm.ops.selective_scan_interface
                      .selective_scan_fn` with D, no z, delta already through softplus):
                      out = scan(u, delta, A = -exp(A_log), B, C) + u * D, from a zero state.
                      Its arithmetic is the Mamba-1 block's kernels' own thread bodies.
"""
from std.math import isfinite
from sequence.exec_trait import Exec
from sequence.ops import Args, FP, OP_FILL
from autograd.tape import Tape, Tensor, BackwardRules
from autograd.rankn import ShapeN
from autograd.transformer_primitives_ops import OP_TFP_LINEAR
from autograd.ssm_ops import (
    OP_SSM_M1_A,
    OP_SSM_M1_SCAN,
    OP_SSM_M1_CKPT,
    OP_SSM_M1_BWD,
    OP_SSM_M1_DBC,
    OP_SSM_M1_DAP,
    OP_SSM_M1_FOLD,
    OP_SSM_M1_DALOG,
    OP_SSM_M1_DU,
    ssm_m1_states_supported,
)
from mamba.host.gen.gemm_identical_parts import contract_partition
# Mamba-3 (lane neural-blocks-3-m3): the trapezoid core, tag 242
from autograd.ssm_m3_primitives import GRAPH_TRAPEZOID_SCAN, trapezoid_scan_bwd
from gemm.experiments.neural_profile import NEURAL_PROFILE_CHANGED, NEURAL_LEAF, neural_partition

# Tags clear of the import family (192..207) and the transformer primitives (208..211).
comptime GRAPH_SELECTIVE_SCAN = 240
# lane neural-blocks-3: 241 the Mamba-2 SSD core, 242 the Mamba-3 trapezoid core (layer codes 43, 44; lm_head 45)
comptime SSM_TAG_FIRST = 240
comptime SSM_TAG_LAST = 247


def _fill[E: Exec](mut ex: E, p: FP, n: Int, value: Float32) raises:
    var a = Args()
    a.p0 = p
    a.f0 = value
    ex.launch[OP_FILL](a, n)


def graph_selective_scan[E: Exec](
    mut tape: Tape, mut ex: E, u: Tensor, delta: Tensor, bm: Tensor, cm: Tensor, a_log: Tensor, d_skip: Tensor,
) raises -> Tensor:
    """The Mamba-1 selective scan over (batch, length) sequences.

    u, delta: (..., length, dim); bm, cm: (..., length, state); a_log: (dim, state); d_skip: dim values.
    Returns out (..., length, dim) = y + u * D with y[t] = sum_n C[t, n] h[t, n] and
    h[t] = exp(delta[t] A) h[t-1] + delta[t] B[t] u[t], A = -exp(A_log), h[-1] = 0.

    Serial by nature over time; each (sequence, channel) chain is one element (one GPU thread), as in the
    Mamba-1 block."""
    var shape = tape.shape_of(u)
    if shape.ndim < 2:
        raise Error("ValueError: selective_scan: u needs (length, dim) axes")
    var m = tape.rows(u)
    var dim = tape.cols(u)
    var length = shape.d[shape.ndim - 2]
    var n = tape.cols(bm)
    if length < 1 or m % length != 0:
        raise Error("ValueError: selective_scan: the rows of u must be whole sequences")
    var batch = m // length
    if tape.rows(delta) != m or tape.cols(delta) != dim:
        raise Error("ValueError: selective_scan: delta must have the shape of u")
    if not ssm_m1_states_supported(n):
        raise Error("ValueError: selective_scan: the state size must be 4, 8 or 16")
    if tape.rows(bm) != m or tape.rows(cm) != m or tape.cols(cm) != n:
        raise Error("ValueError: selective_scan: B and C must be (rows of u, state)")
    if tape.rows(a_log) != dim or tape.cols(a_log) != n:
        raise Error("ValueError: selective_scan: A_log must be (dim, state)")
    if tape.rows(d_skip) * tape.cols(d_skip) != dim:
        raise Error("ValueError: selective_scan: D must hold one value per channel")
    var up = tape.value_of(ex, u)
    var dp = tape.value_of(ex, delta)
    var bp = tape.value_of(ex, bm)
    var cp = tape.value_of(ex, cm)
    var alp = tape.value_of(ex, a_log)
    var skp = tape.value_of(ex, d_skip)
    var a_buf = tape.reserve(ex, dim * n)
    var h = tape.reserve(ex, batch * dim * n)
    var y = tape.reserve(ex, m * dim)
    var output = tape.reserve(ex, m * dim)
    var a = Args()
    a.p0 = alp
    a.p1 = a_buf
    ex.launch[OP_SSM_M1_A](a, dim * n)
    _fill(ex, h, batch * dim * n, Float32(0))
    var s = Args()
    s.p0 = output
    s.p1 = y
    s.p2 = h
    s.p3 = up
    s.p4 = dp
    s.p5 = a_buf
    s.p6 = bp
    s.p7 = cp
    s.p8 = skp
    s.i0 = batch
    s.i1 = length
    s.i2 = dim
    s.i3 = n
    ex.launch[OP_SSM_M1_SCAN](s, batch * dim)
    var saved = Args()
    saved.i0 = batch
    saved.i1 = length
    saved.i2 = dim
    saved.i3 = n
    var part = neural_partition[NEURAL_LEAF](dim) if NEURAL_PROFILE_CHANGED else contract_partition(dim)
    saved.i4 = part[0]
    saved.i5 = part[1]
    saved.p0 = a_buf
    saved.p1 = h
    if tape.is_recording():
        # backward storage, the Mamba-1 block's scan scratch for one layer (autograd/rules_mamba.mojo)
        saved.p2 = tape.reserve(ex, batch * (length + 1) * dim * n)  # h at every token
        saved.p3 = tape.reserve(ex, m * dim * n)  # dh
        saved.p4 = tape.reserve(ex, m * dim)  # du (scan part)
        saved.p5 = tape.reserve(ex, m * dim)  # w = delta * u
        saved.p6 = tape.reserve(ex, m * dim)  # pD = dsk * u
        saved.p7 = tape.reserve(ex, batch * dim * n)  # dA partials
        saved.p8 = tape.reserve(ex, dim * n)  # dA
        saved.p9 = tape.reserve(ex, m)  # ones (dD = pD^T ones)
        _fill(ex, saved.p9, m, Float32(1))
    else:
        saved.p2 = a_buf
        saved.p3 = a_buf
        saved.p4 = a_buf
        saved.p5 = a_buf
        saved.p6 = a_buf
        saved.p7 = a_buf
        saved.p8 = a_buf
        saved.p9 = a_buf
    return tape.custom_nd(ex, [u, delta, bm, cm, a_log, d_skip], output, shape, GRAPH_SELECTIVE_SCAN, saved)


def _selective_scan_bwd[E: Exec](mut ex: E, saved: Args, up: FP, inputs: List[FP], gradients: List[FP]) raises:
    var batch = saved.i0
    var length = saved.i1
    var dim = saved.i2
    var n = saved.i3
    var m = batch * length
    var a_buf = saved.p0
    var h0 = saved.p1
    var hck = saved.p2
    var dh = saved.p3
    var du_s = saved.p4
    var w = saved.p5
    var pd = saved.p6
    var dap = saved.p7
    var da = saved.p8
    var ones = saved.p9
    var u = inputs[0]
    var delta = inputs[1]
    var bm = inputs[2]
    var cm = inputs[3]
    # h at every token from the zero state (the forward left its final state in h0)
    _fill(ex, h0, batch * dim * n, Float32(0))
    var c = Args()
    c.p0 = hck
    c.p1 = h0
    c.p2 = u
    c.p3 = delta
    c.p4 = a_buf
    c.p5 = bm
    c.i0 = batch
    c.i1 = length
    c.i2 = dim
    c.i3 = n
    ex.launch[OP_SSM_M1_CKPT](c, batch * dim)
    # T1, B17, B18: dh, du (scan part), ddelta (into its gradient), w; the scan's dy is out's upstream
    var b = Args()
    b.p0 = dh
    b.p1 = du_s
    b.p2 = gradients[1]
    b.p3 = w
    b.p4 = up
    b.p5 = cm
    b.p6 = bm
    b.p7 = delta
    b.p8 = a_buf
    b.p9 = u
    b.p10 = hck
    b.i0 = batch
    b.i1 = length
    b.i2 = dim
    b.i3 = n
    ex.launch[OP_SSM_M1_BWD](b, batch * dim)
    var j = Args()
    j.p0 = up
    j.p1 = inputs[5]
    j.p2 = u
    j.p3 = du_s
    j.p4 = gradients[0]
    j.p5 = pd
    j.i0 = dim
    ex.launch[OP_SSM_M1_DU](j, m * dim)
    var t3 = Args()
    t3.p0 = gradients[3]
    t3.p1 = gradients[2]
    t3.p2 = up
    t3.p3 = hck
    t3.p4 = w
    t3.p5 = dh
    t3.i0 = batch
    t3.i1 = length
    t3.i2 = dim
    t3.i3 = n
    t3.i4 = saved.i4
    t3.i5 = saved.i5
    ex.launch[OP_SSM_M1_DBC](t3, m * n)
    var p = Args()
    p.p0 = dap
    p.p1 = dh
    p.p2 = delta
    p.p3 = a_buf
    p.p4 = u
    p.p5 = bm
    p.p6 = hck
    p.i0 = batch
    p.i1 = length
    p.i2 = dim
    p.i3 = n
    ex.launch[OP_SSM_M1_DAP](p, batch * dim)
    var f = Args()
    f.p0 = da
    f.p1 = dap
    f.i0 = batch
    f.i1 = dim * n
    ex.launch[OP_SSM_M1_FOLD](f, dim * n)
    var g = Args()
    g.p0 = gradients[4]
    g.p1 = da
    g.p2 = a_buf
    ex.launch[OP_SSM_M1_DALOG](g, dim * n)
    # dD = pD^T ones through the neural GEMM cell (the transformer primitives' projection op), the gemm
    # contract's leaves over the M rows
    var r = Args()
    r.p0 = pd
    r.p1 = ones
    r.p2 = gradients[5]
    r.i0 = dim
    r.i1 = 1
    r.i2 = m
    r.i3 = 1
    r.i4 = dim
    r.i5 = 1
    r.i6 = 0
    ex.launch[OP_TFP_LINEAR](r, dim)


@fieldwise_init
struct SsmRules(BackwardRules, ImplicitlyCopyable, Movable):
    """The backward of the state-space tape operations (tags SSM_TAG_FIRST..SSM_TAG_LAST)."""

    def supports(self, tag: Int) -> Bool:
        return tag == GRAPH_SELECTIVE_SCAN or tag == GRAPH_TRAPEZOID_SCAN  # Mamba-3 (lane neural-blocks-3-m3)

    def zeroed_destination(self, tag: Int, saved: Args, edge: Int) -> Bool:
        # every destination is overwritten whole: du, ddelta (M dim), dB, dC (M state), dA_log, dD
        return False

    def writes_total(self, tag: Int, saved: Args, edge: Int) -> Bool:
        return False

    def apply[E: Exec](
        self, mut ex: E, tag: Int, saved: Args, output: FP, upstream: FP,
        inputs: List[FP], gradients: List[FP], rows: List[Int], cols: List[Int],
    ) raises:
        if tag == GRAPH_SELECTIVE_SCAN:
            _selective_scan_bwd(ex, saved, upstream, inputs, gradients)
        # Mamba-3 (lane neural-blocks-3-m3)
        elif tag == GRAPH_TRAPEZOID_SCAN:
            trapezoid_scan_bwd(ex, saved, upstream, inputs, gradients)
        else:
            raise Error("unknown state-space backward rule")
