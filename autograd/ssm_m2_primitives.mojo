# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Mamba-2 SSD core as a generic differentiable tape operation (lane neural-blocks-3, 2026-11-22).

graph_ssd_scan  the Mamba-2 chunked state-space dual scan (`mamba_ssm.modules.ssd_minimal`, discretize
                first, profile `mojolearn.identical.mamba2.fp32.v1`, `mamba/IDENTICAL_MAMBA2_CONTRACT.md`) with
                the D skip, from a zero state: per head h (P = 64 channels, one group of N = 128 states)
                    h[t] = exp(dt[t, h] A[h]) h[t-1] + dt[t, h] B[t] x[t, h]^T,   A = -exp(A_log)
                    out[t, h, p] = C[t] . h[t][p] + D[h] x[t, h, p]
                computed by the block's chunked algorithm (chunk Q = 256, S10-S20). Its arithmetic is the
                Mamba-2 block's kernels' own thread bodies (`autograd/kernels/mamba2_ssd_body.mojo`),
                launched element by element through `autograd/ssm_m2_ops.mojo` on any executor; the
                backward is the block's SSD backward, launch for launch (`autograd/rules_mamba.mojo::
                _m2_ssd_bwd`), stopping at dt (the block continues through its softplus).
"""
from sequence.exec_trait import Exec
from sequence.ops import Args, FP, OP_FILL
from autograd.tape import Tape, Tensor
from autograd.rankn_api import rn_dims, rn_shape
from autograd.transformer_primitives_ops import OP_TFP_LINEAR
from mamba.checks.mamba2_fixture import M2_D_STATE, M2_HEADDIM
from autograd.kernels.mamba2_ssd_body import m2_q_eff, m2_fold_rows_per_tile, m2_fold_tiles
from autograd.ssm_m2_ops import (
    OP_SSM_M2_DISC,
    OP_SSM_M2_CUMSUM,
    OP_SSM_M2_SEGL,
    OP_SSM_M2_CBG,
    OP_SSM_M2_YDIAG,
    OP_SSM_M2_DECAY,
    OP_SSM_M2_CSTATE,
    OP_SSM_M2_STATEPASS,
    OP_SSM_M2_YOFF,
    OP_SSM_M2_SKIP,
    OP_SSM_M2_DSKIP,
    OP_SSM_M2_S18_DPASS,
    OP_SSM_M2_S18_DC,
    OP_SSM_M2_REV_STATE,
    OP_SSM_M2_CS_DDECAY,
    OP_SSM_M2_SCALE_DACS,
    OP_SSM_M2_DECAY_DACS,
    OP_SSM_M2_MERGE_DACS,
    OP_SSM_M2_REV_CUMSUM,
    OP_SSM_M2_YD_MATRIX,
    OP_SSM_M2_CB_BWD,
    OP_SSM_M2_CS_DXD_DB,
    OP_SSM_M2_SEG_BWD,
    OP_SSM_M2_DA,
    OP_SSM_M2_FOLD_ROWS,
    OP_SSM_M2_FOLD_TILES,
    OP_SSM_M2_YD_XD,
    OP_SSM_M2_MERGE,
    OP_SSM_M2_PACK,
)
from autograd.ssm_ops import OP_SSM_M1_A

#: the tape tag of the op (SsmRules, autograd/ssm_primitives.mojo)
comptime GRAPH_SSD_SCAN = 241
#: the head width and the state size the profile fixes
comptime SSD_HEADDIM = M2_HEADDIM
comptime SSD_STATES = M2_D_STATE


@fieldwise_init
struct _Dims(ImplicitlyCopyable, Movable):
    var b: Int
    var t: Int
    var nh: Int
    var nc: Int
    var q: Int

    def m(self) -> Int:
        return self.b * self.t

    def di(self) -> Int:
        return self.nh * SSD_HEADDIM

    def cd(self) -> Int:
        return self.di() + 2 * SSD_STATES

    def s(self) -> Int:
        return self.b * self.nc * self.nh * SSD_HEADDIM * SSD_STATES

    def h0(self) -> Int:
        return self.b * self.nh * SSD_HEADDIM * SSD_STATES

    def r(self) -> Int:
        return self.b * self.nc * self.nh


def _dims(saved: Args) -> _Dims:
    return _Dims(saved.i0, saved.i1, saved.i2, saved.i3, saved.i4)


# The forward block (saved.p0), in order: a, xd, da, dacs, seg_l, cb_g, ydiag, decay, cstate, pass_states,
# h_last, h_state, yoff, y.
def _fwd_sizes(d: _Dims) -> List[Int]:
    var m = d.m()
    var rq = d.r() * d.q
    var out: List[Int] = [
        d.nh, m * d.di(), m * d.nh, rq, d.r() * d.q * d.q, d.b * d.nc * d.q * d.q, m * d.di(), rq, d.s(), d.s(),
        d.h0(), d.h0(), m * d.di(), m * d.di(),
    ]
    return out^


# The backward scratch (saved.p1), in order.
comptime _B_D_SCAN = 0
comptime _B_DX_D = 1
comptime _B_DD_PROD = 2
comptime _B_ONES_M = 3
comptime _B_DIRECT = 4
comptime _B_DC_YOFF = 5
comptime _B_DDACS_YOFF = 6
comptime _B_DPASS = 7
comptime _B_DCSTATE = 8
comptime _B_DSCALE_PROD = 9
comptime _B_DINITIAL = 10
comptime _B_DFINAL = 11
comptime _B_DDECAY_CS = 12
comptime _B_DSCALE = 13
comptime _B_ONES_PN = 14
comptime _B_DDACS_STATE = 15
comptime _B_DDACS_TOTAL = 16
comptime _B_DDACS_DECAY = 17
comptime _B_DDA = 18
comptime _B_DDT = 19
comptime _B_DXD_YD = 20
comptime _B_DX_XD = 21
comptime _B_DDT_XD = 22
comptime _B_DCB_YD = 23
comptime _B_DSEG_YD = 24
comptime _B_DB_CB = 25
comptime _B_DC_CB = 26
comptime _B_DXD_CS = 27
comptime _B_DXD_TOTAL = 28
comptime _B_DB_CS = 29
comptime _B_DB_TOTAL = 30
comptime _B_DC_TOTAL = 31
comptime _B_DX_TOTAL = 32
comptime _B_DDA_SEG = 33
comptime _B_DDA_TOTAL = 34
comptime _B_FOLD_PART = 35
comptime _B_DA = 36


def _bwd_sizes(d: _Dims) -> List[Int]:
    var m = d.m()
    var di = d.di()
    var n = SSD_STATES
    var rq = d.r() * d.q
    var tiles = m2_fold_tiles(m)
    var out: List[Int] = [
        m * di, m * di, m * d.nh, m,
        d.s(), d.b * d.nc * d.q * n, rq, d.s(), d.s(), d.s(), d.h0(), d.h0(),
        rq, d.r(), SSD_HEADDIM * n, rq, rq, rq,
        m * d.nh, m * d.nh, m * di, m * di, m * d.nh, d.b * d.nc * d.q * d.q, d.r() * d.q * d.q,
        m * n, m * n, m * di, m * di, m * n, m * n, m * n, m * di,
        m * d.nh, m * d.nh, tiles * d.nh, d.nh,
    ]
    return out^


def _offsets(sizes: List[Int]) -> List[Int]:
    var out = List[Int](capacity=len(sizes) + 1)
    var o = 0
    for i in range(len(sizes)):  # small-loop(at most 37 scratch buffers of one layer)
        out.append(o)
        o += sizes[i]
    out.append(o)
    return out^


def _fill[E: Exec](mut ex: E, p: FP, n: Int, value: Float32) raises:
    var a = Args()
    a.p0 = p
    a.f0 = value
    ex.launch[OP_FILL](a, n)


def _dims_args(mut a: Args, d: _Dims):
    # the S10-S19 integer block most bodies take: B, T, H, DI, CD, C, Q
    a.i0 = d.b
    a.i1 = d.t
    a.i2 = d.nh
    a.i3 = d.di()
    a.i4 = d.cd()
    a.i5 = d.nc
    a.i6 = d.q


def _bthcq(mut a: Args, d: _Dims):
    # B, T, H, C, Q
    a.i0 = d.b
    a.i1 = d.t
    a.i2 = d.nh
    a.i3 = d.nc
    a.i4 = d.q


def graph_ssd_scan[E: Exec](
    mut tape: Tape, mut ex: E, xbc: Tensor, dt: Tensor, a_log: Tensor, d_skip: Tensor,
) raises -> Tensor:
    """The Mamba-2 SSD core over (batch, length) sequences.

    xbc: (..., length, H * 64 + 256) = [x | B | C] per row; dt: (..., length, H), already positive (for
    example after softplus); a_log, d_skip: H values. Returns (..., length, H * 64): y + D x.

    Each of the block's stages is one element op (the block's own thread bodies): chunk-local stages run in
    parallel over every (sequence, chunk, head) cell; S17 walks the chunks serially per state cell, as the
    block does."""
    var shape = tape.shape_of(xbc)
    if shape.ndim < 2:
        raise Error("ValueError: ssd_scan: x needs (length, features) axes")
    var m = tape.rows(xbc)
    var cd = tape.cols(xbc)
    var t = shape.d[shape.ndim - 2]
    if t < 1 or m % t != 0:
        raise Error("ValueError: ssd_scan: the rows must be whole sequences")
    var nh = tape.cols(dt)
    if nh < 1 or cd != nh * SSD_HEADDIM + 2 * SSD_STATES:
        raise Error("ValueError: ssd_scan: x must be (rows, heads * 64 + 256) for dt's heads")
    if tape.rows(dt) != m:
        raise Error("ValueError: ssd_scan: dt must have the rows of x")
    if tape.rows(a_log) * tape.cols(a_log) != nh or tape.rows(d_skip) * tape.cols(d_skip) != nh:
        raise Error("ValueError: ssd_scan: A_log and D must hold one value per head")
    var q = m2_q_eff()
    var nc = (t + q - 1) // q
    var d = _Dims(m // t, t, nh, nc, q)
    var xp = tape.value_of(ex, xbc)
    var dtp = tape.value_of(ex, dt)
    var alp = tape.value_of(ex, a_log)
    var skp = tape.value_of(ex, d_skip)
    var fs = _fwd_sizes(d)
    var fo = _offsets(fs)
    var base = tape.reserve(ex, fo[len(fs)])
    var a_buf = base + fo[0]
    var xd = base + fo[1]
    var da = base + fo[2]
    var dacs = base + fo[3]
    var seg_l = base + fo[4]
    var cb_g = base + fo[5]
    var ydiag = base + fo[6]
    var decay = base + fo[7]
    var cstate = base + fo[8]
    var pass_states = base + fo[9]
    var h_last = base + fo[10]
    var h_state = base + fo[11]
    var yoff = base + fo[12]
    var y = base + fo[13]
    var output = tape.reserve(ex, m * d.di())
    # S5: A = -exp(A_log), the block's (and Mamba-1's) body
    var a5 = Args()
    a5.p0 = alp
    a5.p1 = a_buf
    ex.launch[OP_SSM_M1_A](a5, nh)
    _fill(ex, h_state, d.h0(), Float32(0))
    var s10 = Args()
    s10.p0 = xd
    s10.p1 = da
    s10.p2 = xp
    s10.p3 = dtp
    s10.p4 = a_buf
    s10.i0 = m
    s10.i1 = nh
    s10.i2 = cd
    ex.launch[OP_SSM_M2_DISC](s10, m * nh)
    var s11 = Args()
    s11.p0 = dacs
    s11.p1 = da
    _bthcq(s11, d)
    ex.launch[OP_SSM_M2_CUMSUM](s11, d.r())
    var s12 = Args()
    s12.p0 = seg_l
    s12.p1 = da
    _bthcq(s12, d)
    ex.launch[OP_SSM_M2_SEGL](s12, d.r() * q)
    var g = Args()
    g.p0 = cb_g
    g.p1 = xp
    g.i0 = d.b
    g.i1 = t
    g.i2 = d.di()
    g.i3 = cd
    g.i4 = nc
    g.i5 = q
    ex.launch[OP_SSM_M2_CBG](g, d.b * nc * q * q)
    var s14 = Args()
    s14.p0 = ydiag
    s14.p1 = cb_g
    s14.p2 = seg_l
    s14.p3 = xd
    s14.p4 = dtp
    _bthcq(s14, d)
    ex.launch[OP_SSM_M2_YDIAG](s14, m * d.di())
    var s15 = Args()
    s15.p0 = decay
    s15.p1 = dacs
    s15.i0 = d.r() * q
    s15.i1 = q
    ex.launch[OP_SSM_M2_DECAY](s15, d.r() * q)
    var s16 = Args()
    s16.p0 = cstate
    s16.p1 = xp
    s16.p2 = decay
    s16.p3 = xd
    s16.p4 = dtp
    s16.i0 = d.b
    s16.i1 = t
    s16.i2 = nh
    s16.i3 = d.di()
    s16.i4 = cd
    s16.i5 = nc
    s16.i6 = q
    ex.launch[OP_SSM_M2_CSTATE](s16, d.s())
    var s17 = Args()
    s17.p0 = pass_states
    s17.p1 = h_last
    s17.p2 = h_state
    s17.p3 = cstate
    s17.p4 = dacs
    s17.i0 = d.b
    s17.i1 = nh
    s17.i2 = nc
    s17.i3 = q
    s17.i4 = 0  # no resumption boundary: the op starts from zero and keeps no state
    ex.launch[OP_SSM_M2_STATEPASS](s17, d.h0())
    var s18 = Args()
    s18.p0 = yoff
    s18.p1 = y
    s18.p2 = ydiag
    s18.p3 = xp
    s18.p4 = pass_states
    s18.p5 = dacs
    _dims_args(s18, d)
    ex.launch[OP_SSM_M2_YOFF](s18, m * d.di())
    var s20 = Args()
    s20.p0 = output
    s20.p1 = y
    s20.p2 = xp
    s20.p3 = skp
    s20.i0 = d.b
    s20.i1 = t
    s20.i2 = 0
    s20.i3 = nh
    s20.i4 = cd
    ex.launch[OP_SSM_M2_SKIP](s20, m * d.di())
    var saved = Args()
    saved.i0 = d.b
    saved.i1 = t
    saved.i2 = nh
    saved.i3 = nc
    saved.i4 = q
    saved.p0 = base
    saved.p1 = base
    if tape.is_recording():
        var bs = _bwd_sizes(d)
        var bo = _offsets(bs)
        saved.p1 = tape.reserve(ex, bo[len(bs)])
    var od = rn_dims(shape)
    od[shape.ndim - 1] = d.di()
    return tape.custom_nd(ex, [xbc, dt, a_log, d_skip], output, rn_shape(od, "ssd_scan"), GRAPH_SSD_SCAN, saved)


def ssd_scan_backward[E: Exec](mut ex: E, saved: Args, up: FP, inputs: List[FP], gradients: List[FP]) raises:
    """The block's SSD backward (`_m2_ssd_bwd`), launch for launch, into the op's four gradients:
    gradients[0] xbc ([dx | dB | dC]), [1] dt, [2] A_log, [3] D."""
    var d = _dims(saved)
    var m = d.m()
    var di = d.di()
    var cd = d.cd()
    var nh = d.nh
    var q = d.q
    var n = SSD_STATES
    var fo = _offsets(_fwd_sizes(d))
    var bo = _offsets(_bwd_sizes(d))
    var fb = saved.p0
    var bb = saved.p1
    var a_buf = fb + fo[0]
    var xd = fb + fo[1]
    var dacs = fb + fo[3]
    var seg_l = fb + fo[4]
    var cb_g = fb + fo[5]
    var decay = fb + fo[7]
    var pass_states = fb + fo[9]
    var xp = inputs[0]
    var dtp = inputs[1]
    var skp = inputs[3]
    var d_scan = bb + bo[_B_D_SCAN]
    var dx_d = bb + bo[_B_DX_D]
    var dd_prod = bb + bo[_B_DD_PROD]
    var ones_m = bb + bo[_B_ONES_M]
    var direct = bb + bo[_B_DIRECT]
    var dc_yoff = bb + bo[_B_DC_YOFF]
    var ddacs_yoff = bb + bo[_B_DDACS_YOFF]
    var dpass = bb + bo[_B_DPASS]
    var dcstate = bb + bo[_B_DCSTATE]
    var dscale_prod = bb + bo[_B_DSCALE_PROD]
    var dinitial = bb + bo[_B_DINITIAL]
    var dfinal = bb + bo[_B_DFINAL]
    var ddecay_cs = bb + bo[_B_DDECAY_CS]
    var dscale = bb + bo[_B_DSCALE]
    var ones_pn = bb + bo[_B_ONES_PN]
    var ddacs_state = bb + bo[_B_DDACS_STATE]
    var ddacs_total = bb + bo[_B_DDACS_TOTAL]
    var ddacs_decay = bb + bo[_B_DDACS_DECAY]
    var dda = bb + bo[_B_DDA]
    var ddt = bb + bo[_B_DDT]
    var dxd_yd = bb + bo[_B_DXD_YD]
    var dx_xd = bb + bo[_B_DX_XD]
    var ddt_xd = bb + bo[_B_DDT_XD]
    var dcb_yd = bb + bo[_B_DCB_YD]
    var dseg_yd = bb + bo[_B_DSEG_YD]
    var db_cb = bb + bo[_B_DB_CB]
    var dc_cb = bb + bo[_B_DC_CB]
    var dxd_cs = bb + bo[_B_DXD_CS]
    var dxd_total = bb + bo[_B_DXD_TOTAL]
    var db_cs = bb + bo[_B_DB_CS]
    var db_total = bb + bo[_B_DB_TOTAL]
    var dc_total = bb + bo[_B_DC_TOTAL]
    var dx_total = bb + bo[_B_DX_TOTAL]
    var dda_seg = bb + bo[_B_DDA_SEG]
    var dda_total = bb + bo[_B_DDA_TOTAL]
    var fold_part = bb + bo[_B_FOLD_PART]
    var d_a = bb + bo[_B_DA]
    _fill(ex, ones_m, m, Float32(1))
    _fill(ex, ones_pn, SSD_HEADDIM * n, Float32(1))
    _fill(ex, dfinal, d.h0(), Float32(0))
    _fill(ex, ddacs_state, d.r() * q, Float32(0))
    # C4: the D skip (S20) backward; dD = d_d_product^T ones through the neural GEMM cell
    var k = Args()
    k.p0 = d_scan
    k.p1 = dx_d
    k.p2 = dd_prod
    k.p3 = up
    k.p4 = xp
    k.p5 = skp
    k.i0 = m * nh
    k.i1 = di
    k.i2 = cd
    k.i3 = SSD_HEADDIM
    ex.launch[OP_SSM_M2_DSKIP](k, m * nh)
    var r = Args()
    r.p0 = dd_prod
    r.p1 = ones_m
    r.p2 = gradients[3]
    r.i0 = nh
    r.i1 = 1
    r.i2 = m
    r.i3 = 1
    r.i4 = nh
    r.i5 = 1
    r.i6 = 0
    ex.launch[OP_TFP_LINEAR](r, nh)
    # S18 direct d_pass, then dC and the Y_off part of d_dacs
    var p1 = Args()
    p1.p0 = direct
    p1.p1 = d_scan
    p1.p2 = xp
    p1.p3 = dacs
    _dims_args(p1, d)
    ex.launch[OP_SSM_M2_S18_DPASS](p1, d.s())
    var p2 = Args()
    p2.p0 = dc_yoff
    p2.p1 = ddacs_yoff
    p2.p2 = d_scan
    p2.p3 = xp
    p2.p4 = pass_states
    p2.p5 = dacs
    _dims_args(p2, d)
    var extra = d.r() * q
    if m * n > extra:
        extra = m * n
    ex.launch[OP_SSM_M2_S18_DC](p2, extra)
    # S17 reversed
    var rv = Args()
    rv.p0 = dpass
    rv.p1 = dcstate
    rv.p2 = dscale_prod
    rv.p3 = dinitial
    rv.p4 = direct
    rv.p5 = dfinal
    rv.p6 = pass_states
    rv.p7 = dacs
    rv.i0 = d.b
    rv.i1 = nh
    rv.i2 = d.nc
    rv.i3 = q
    ex.launch[OP_SSM_M2_REV_STATE](rv, d.h0())
    var cdd = Args()
    cdd.p0 = ddecay_cs
    cdd.p1 = dcstate
    cdd.p2 = xd
    cdd.p3 = xp
    _dims_args(cdd, d)
    ex.launch[OP_SSM_M2_CS_DDECAY](cdd, d.r() * q)
    # DEVIATION 1345's P*N contraction of the scale products, then the d_dacs parts and their merge
    var sp = Args()
    sp.p0 = dscale_prod
    sp.p1 = ones_pn
    sp.p2 = dscale
    sp.i0 = d.r()
    sp.i1 = 1
    sp.i2 = SSD_HEADDIM * n
    sp.i3 = SSD_HEADDIM * n
    sp.i4 = 1
    sp.i5 = 1
    sp.i6 = 0
    ex.launch[OP_TFP_LINEAR](sp, d.r())
    var sd = Args()
    sd.p0 = ddacs_state
    sd.p1 = dscale
    sd.p2 = dacs
    sd.i0 = d.b
    sd.i1 = d.nc
    sd.i2 = nh
    sd.i3 = q
    ex.launch[OP_SSM_M2_SCALE_DACS](sd, d.r())
    var dd = Args()
    dd.p0 = ddacs_decay
    dd.p1 = ddecay_cs
    dd.p2 = decay
    dd.i0 = d.r()
    dd.i1 = q
    ex.launch[OP_SSM_M2_DECAY_DACS](dd, d.r())
    var md = Args()
    md.p0 = ddacs_total
    md.p1 = ddacs_yoff
    md.p2 = ddacs_state
    md.p3 = ddacs_decay
    md.i0 = d.r() * q
    ex.launch[OP_SSM_M2_MERGE_DACS](md, d.r() * q)
    # S11 reversed
    var rc = Args()
    rc.p0 = dda
    rc.p1 = ddacs_total
    _bthcq(rc, d)
    ex.launch[OP_SSM_M2_REV_CUMSUM](rc, d.r())
    # S13/S14 reversed: dG, dL; dB, dC from dG; dX_d and dB from the states; dL into d_dA
    var ym = Args()
    ym.p0 = dcb_yd
    ym.p1 = dseg_yd
    ym.p2 = d_scan
    ym.p3 = xd
    ym.p4 = cb_g
    ym.p5 = seg_l
    _bthcq(ym, d)
    ex.launch[OP_SSM_M2_YD_MATRIX](ym, d.b * d.nc * q * q)
    var cb = Args()
    cb.p0 = db_cb
    cb.p1 = dc_cb
    cb.p2 = dcb_yd
    cb.p3 = xp
    cb.i0 = d.b
    cb.i1 = d.t
    cb.i2 = di
    cb.i3 = cd
    cb.i4 = d.nc
    cb.i5 = q
    ex.launch[OP_SSM_M2_CB_BWD](cb, m * n)
    var cx = Args()
    cx.p0 = dxd_cs
    cx.p1 = db_cs
    cx.p2 = db_total
    cx.p3 = db_cb
    cx.p4 = dcstate
    cx.p5 = xd
    cx.p6 = xp
    cx.p7 = decay
    _dims_args(cx, d)
    var cxn = m * di
    if m * n > cxn:
        cxn = m * n
    ex.launch[OP_SSM_M2_CS_DXD_DB](cx, cxn)
    var sg = Args()
    sg.p0 = dda_seg
    sg.p1 = dda_total
    sg.p2 = dseg_yd
    sg.p3 = seg_l
    sg.p4 = dda
    _bthcq(sg, d)
    ex.launch[OP_SSM_M2_SEG_BWD](sg, m * nh)
    # S10 reversed for dA: d_dt's dA part per cell, then dA folded over the rows (row tiles), dA_log = dA * A
    var dc = Args()
    dc.p0 = ddt
    dc.p1 = dda_total
    dc.p2 = a_buf
    dc.i0 = m
    dc.i1 = nh
    ex.launch[OP_SSM_M2_DA](dc, m * nh)
    var per = m2_fold_rows_per_tile(m)
    var tiles = m2_fold_tiles(m)
    var fr = Args()
    fr.p0 = fold_part
    fr.p1 = dtp
    fr.p2 = dda_total
    fr.i0 = m
    fr.i1 = nh
    fr.i2 = per
    fr.i3 = tiles
    fr.i4 = 1
    ex.launch[OP_SSM_M2_FOLD_ROWS](fr, tiles * nh)
    var ft = Args()
    ft.p0 = d_a
    ft.p1 = d_a
    ft.p2 = gradients[2]
    ft.p3 = a_buf
    ft.p4 = fold_part
    ft.i0 = tiles
    ft.i1 = nh
    ft.i2 = nh
    ft.i3 = 1
    ex.launch[OP_SSM_M2_FOLD_TILES](ft, nh)
    # S10 reversed for X_d: dx and dt (dt's gradient is written whole into gradients[1])
    var yx = Args()
    yx.p0 = dxd_yd
    yx.p1 = dx_xd
    yx.p2 = ddt_xd
    yx.p3 = gradients[1]
    yx.p4 = d_scan
    yx.p5 = cb_g
    yx.p6 = seg_l
    yx.p7 = xp
    yx.p8 = dtp
    yx.p9 = ddt
    yx.p10 = dxd_cs
    yx.p11 = dxd_total
    _dims_args(yx, d)
    ex.launch[OP_SSM_M2_YD_XD](yx, m * nh)
    # the dC and dx joins, then [dx | dB | dC] into the xbc gradient
    var mg = Args()
    mg.p0 = dc_total
    mg.p1 = dx_total
    mg.p2 = dc_yoff
    mg.p3 = dc_cb
    mg.p4 = dx_d
    mg.p5 = dx_xd
    mg.i0 = m * n
    mg.i1 = m * di
    var mgn = m * di
    if m * n > mgn:
        mgn = m * n
    ex.launch[OP_SSM_M2_MERGE](mg, mgn)
    var pk = Args()
    pk.p0 = gradients[0]
    pk.p1 = dx_total
    pk.p2 = db_total
    pk.p3 = dc_total
    pk.i0 = di
    pk.i1 = cd
    ex.launch[OP_SSM_M2_PACK](pk, m * cd)
