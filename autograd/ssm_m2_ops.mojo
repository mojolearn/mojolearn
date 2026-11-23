# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Element functions of the Mamba-2 SSD core as a generic tape operation (lane neural-blocks-3, 2026-11-22).

Read by both executors through `sequence/dispatch.mojo::apply` -> `autograd/ssm_ops.mojo::apply_ssm` (HostExec:
a host loop over the elements; DeviceExec: one GPU thread per element), so the host and the device column run
the same function on the same words. Every element is ONE CALL of the Mamba-2 block kernel's own thread body
(`autograd/kernels/mamba2_ssd_body.mojo`); no SSD arithmetic is written in this file. The argument order of
each op is its body's (pointers p0.., integers i0..), documented beside the op.

  op                    elements                 body (block kernel)
  forward
  OP_SSM_M2_DISC        B T H                    m2_discretize_cell          S10
  OP_SSM_M2_CUMSUM      B H C                    m2_chunk_cumsum_cell        S11
  OP_SSM_M2_SEGL        B C H Q                  m2_seg_l_cell               S12 L
  OP_SSM_M2_CBG         B C Q Q                  m2_cb_g_cell                S12 G
  OP_SSM_M2_YDIAG       B T H P                  m2_ydiag_cell               S13 + S14
  OP_SSM_M2_DECAY       B H C Q                  m2_decay_cell               S15
  OP_SSM_M2_CSTATE      B C H P N                m2_cstate_cell              S15 + S16
  OP_SSM_M2_STATEPASS   B H P N                  m2_statepass_cell           S17
  OP_SSM_M2_YOFF        B T H P                  m2_yoff_y_cell              S18 + S19
  OP_SSM_M2_SKIP        B L H P                  m2_skip_cell                S20
  backward
  OP_SSM_M2_DSKIP       M H                      mamba2_d_skip_backward_cell
  OP_SSM_M2_S18_DPASS   B C H P N                mamba2_s18_direct_dpass_cell
  OP_SSM_M2_S18_DC      max(B H C Q, B T N)      mamba2_s18_dc_ddacs_cell
  OP_SSM_M2_REV_STATE   B H P N                  mamba2_reverse_chunk_state_cell
  OP_SSM_M2_CS_DDECAY   B H C Q                  mamba2_cstate_ddecay_cell
  OP_SSM_M2_SCALE_DACS  B C H                    mamba2_scale_to_dacs_cell
  OP_SSM_M2_DECAY_DACS  B C H                    mamba2_decay_to_dacs_cell
  OP_SSM_M2_MERGE_DACS  B C H Q                  mamba2_merge_dacs_cell
  OP_SSM_M2_REV_CUMSUM  B H C                    mamba2_reverse_cumsum_cell
  OP_SSM_M2_YD_MATRIX   B C Q Q                  mamba2_ydiag_matrix_backward_cell
  OP_SSM_M2_CB_BWD      B T N                    mamba2_cb_backward_cell
  OP_SSM_M2_CS_DXD_DB   max(B T H P, B T N)      mamba2_cstate_dxd_db_cell
  OP_SSM_M2_SEG_BWD     B T H                    mamba2_seg_backward_cell
  OP_SSM_M2_DA          B T H                    mamba2_da_cell_body
  OP_SSM_M2_FOLD_ROWS   tiles H                  m2_fold_rows_partial_cell
  OP_SSM_M2_FOLD_TILES  H                        m2_fold_tiles_cell
  OP_SSM_M2_YD_XD       B T H                    mamba2_ydiag_xd_backward_cell
  OP_SSM_M2_MERGE       max(B T H P, B T N)      mamba2_postconv_merge_cell
  OP_SSM_M2_PACK        M (H P + 2 N)            (a copy: [dx | dB | dC] into the xbc gradient's rows)
"""
from sequence.ops import Args, FP
from mamba.checks.mamba2_fixture import M2_D_STATE, M2_HEADDIM
from autograd.kernels.mamba2_ssd_body import (
    m2_discretize_cell,
    m2_chunk_cumsum_cell,
    m2_seg_l_cell,
    m2_cb_g_cell,
    m2_ydiag_cell,
    m2_decay_cell,
    m2_cstate_cell,
    m2_statepass_cell,
    m2_yoff_y_cell,
    m2_skip_cell,
    mamba2_d_skip_backward_cell,
    mamba2_s18_direct_dpass_cell,
    mamba2_s18_dc_ddacs_cell,
    mamba2_reverse_chunk_state_cell,
    mamba2_cstate_ddecay_cell,
    mamba2_scale_to_dacs_cell,
    mamba2_decay_to_dacs_cell,
    mamba2_merge_dacs_cell,
    mamba2_reverse_cumsum_cell,
    mamba2_ydiag_matrix_backward_cell,
    mamba2_cb_backward_cell,
    mamba2_cstate_dxd_db_cell,
    mamba2_seg_backward_cell,
    mamba2_da_cell_body,
    m2_fold_rows_partial_cell,
    m2_fold_tiles_cell,
    mamba2_ydiag_xd_backward_cell,
    mamba2_postconv_merge_cell,
)

comptime OP_SSM_M2_FIRST = 4820
comptime OP_SSM_M2_DISC = 4820
comptime OP_SSM_M2_CUMSUM = 4821
comptime OP_SSM_M2_SEGL = 4822
comptime OP_SSM_M2_CBG = 4823
comptime OP_SSM_M2_YDIAG = 4824
comptime OP_SSM_M2_DECAY = 4825
comptime OP_SSM_M2_CSTATE = 4826
comptime OP_SSM_M2_STATEPASS = 4827
comptime OP_SSM_M2_YOFF = 4828
comptime OP_SSM_M2_SKIP = 4829
comptime OP_SSM_M2_DSKIP = 4830
comptime OP_SSM_M2_S18_DPASS = 4831
comptime OP_SSM_M2_S18_DC = 4832
comptime OP_SSM_M2_REV_STATE = 4833
comptime OP_SSM_M2_CS_DDECAY = 4834
comptime OP_SSM_M2_SCALE_DACS = 4835
comptime OP_SSM_M2_DECAY_DACS = 4836
comptime OP_SSM_M2_MERGE_DACS = 4837
comptime OP_SSM_M2_REV_CUMSUM = 4838
comptime OP_SSM_M2_YD_MATRIX = 4839
comptime OP_SSM_M2_CB_BWD = 4840
comptime OP_SSM_M2_CS_DXD_DB = 4841
comptime OP_SSM_M2_SEG_BWD = 4842
comptime OP_SSM_M2_DA = 4843
comptime OP_SSM_M2_FOLD_ROWS = 4844
comptime OP_SSM_M2_FOLD_TILES = 4845
comptime OP_SSM_M2_YD_XD = 4846
comptime OP_SSM_M2_MERGE = 4847
comptime OP_SSM_M2_PACK = 4848
comptime OP_SSM_M2_LAST = 4859


@always_inline
def _q(p: FP) -> MutPointer[Float32, MutAnyOrigin]:
    # Args carries executor-owned buffers with an untracked origin; the bodies borrow the same live
    # buffers (no owner moves, no lifetime extension at this boundary).
    return p.unsafe_origin_cast[MutAnyOrigin]()


@always_inline
def _i(v: Int) -> Int32:
    return Int32(v)


def apply_ssm_m2[OP: Int](cell: Int, a: Args):
    comptime if OP == OP_SSM_M2_DISC:
        # xd, da, xbc, dt, A; B*T, H, CD
        m2_discretize_cell(cell, _q(a.p0), _q(a.p1), _q(a.p2), _q(a.p3), _q(a.p4), _i(a.i0), _i(a.i1), _i(a.i2))
    elif OP == OP_SSM_M2_CUMSUM:
        # dacs, da; B, T, H, C, Q
        m2_chunk_cumsum_cell(cell, _q(a.p0), _q(a.p1), _i(a.i0), _i(a.i1), _i(a.i2), _i(a.i3), _i(a.i4))
    elif OP == OP_SSM_M2_SEGL:
        # seg_l, da; B, T, H, C, Q
        m2_seg_l_cell(cell, _q(a.p0), _q(a.p1), _i(a.i0), _i(a.i1), _i(a.i2), _i(a.i3), _i(a.i4))
    elif OP == OP_SSM_M2_CBG:
        # cb_g, xbc; B, T, DI, CD, C, Q
        m2_cb_g_cell(cell, _q(a.p0), _q(a.p1), _i(a.i0), _i(a.i1), _i(a.i2), _i(a.i3), _i(a.i4), _i(a.i5))
    elif OP == OP_SSM_M2_YDIAG:
        # ydiag, cb_g, seg_l, xd, dt; B, T, H, C, Q
        m2_ydiag_cell(cell, _q(a.p0), _q(a.p1), _q(a.p2), _q(a.p3), _q(a.p4), _i(a.i0), _i(a.i1), _i(a.i2),
                      _i(a.i3), _i(a.i4))
    elif OP == OP_SSM_M2_DECAY:
        # decay, dacs; n, Q
        m2_decay_cell(cell, _q(a.p0), _q(a.p1), _i(a.i0), _i(a.i1))
    elif OP == OP_SSM_M2_CSTATE:
        # cstate, xbc, decay, xd, dt; B, T, H, DI, CD, C, Q
        m2_cstate_cell(cell, _q(a.p0), _q(a.p1), _q(a.p2), _q(a.p3), _q(a.p4), _i(a.i0), _i(a.i1), _i(a.i2),
                       _i(a.i3), _i(a.i4), _i(a.i5), _i(a.i6))
    elif OP == OP_SSM_M2_STATEPASS:
        # pass_states, h_last, h_state, cstate, dacs; B, H, C, Q, completed chunks
        m2_statepass_cell(cell, _q(a.p0), _q(a.p1), _q(a.p2), _q(a.p3), _q(a.p4), _i(a.i0), _i(a.i1), _i(a.i2),
                          _i(a.i3), _i(a.i4))
    elif OP == OP_SSM_M2_YOFF:
        # yoff, y, ydiag, xbc, pass_states, dacs; B, T, H, DI, CD, C, Q
        m2_yoff_y_cell(cell, _q(a.p0), _q(a.p1), _q(a.p2), _q(a.p3), _q(a.p4), _q(a.p5), _i(a.i0), _i(a.i1),
                       _i(a.i2), _i(a.i3), _i(a.i4), _i(a.i5), _i(a.i6))
    elif OP == OP_SSM_M2_SKIP:
        # out, y, xbc, D; B, L, q0, H, CD
        m2_skip_cell(cell, _q(a.p0), _q(a.p1), _q(a.p2), _q(a.p3), _i(a.i0), _i(a.i1), _i(a.i2), _i(a.i3),
                     _i(a.i4))
    elif OP == OP_SSM_M2_DSKIP:
        # d_scan, d_x_from_d, d_d_product, upstream, xbc, D; M H, DI, CD, P
        mamba2_d_skip_backward_cell(cell, _q(a.p0), _q(a.p1), _q(a.p2), _q(a.p3), _q(a.p4), _q(a.p5),
                                    _i(a.i0), _i(a.i1), _i(a.i2), _i(a.i3))
    elif OP == OP_SSM_M2_S18_DPASS:
        # direct_d_pass, d_y, xbc, dacs; B, T, H, DI, CD, C, Q
        mamba2_s18_direct_dpass_cell(cell, _q(a.p0), _q(a.p1), _q(a.p2), _q(a.p3), _i(a.i0), _i(a.i1),
                                     _i(a.i2), _i(a.i3), _i(a.i4), _i(a.i5), _i(a.i6))
    elif OP == OP_SSM_M2_S18_DC:
        # d_c_yoff, d_dacs_yoff, d_y, xbc, pass_states, dacs; B, T, H, DI, CD, C, Q
        mamba2_s18_dc_ddacs_cell(cell, _q(a.p0), _q(a.p1), _q(a.p2), _q(a.p3), _q(a.p4), _q(a.p5), _i(a.i0),
                                 _i(a.i1), _i(a.i2), _i(a.i3), _i(a.i4), _i(a.i5), _i(a.i6))
    elif OP == OP_SSM_M2_REV_STATE:
        # d_pass, d_cstate, d_scale_product, d_initial, direct_d_pass, d_final, pass_states, dacs; B, H, C, Q
        mamba2_reverse_chunk_state_cell(cell, _q(a.p0), _q(a.p1), _q(a.p2), _q(a.p3), _q(a.p4), _q(a.p5),
                                        _q(a.p6), _q(a.p7), _i(a.i0), _i(a.i1), _i(a.i2), _i(a.i3))
    elif OP == OP_SSM_M2_CS_DDECAY:
        # d_decay, d_cstate, xd, xbc; B, T, H, DI, CD, C, Q
        mamba2_cstate_ddecay_cell(cell, _q(a.p0), _q(a.p1), _q(a.p2), _q(a.p3), _i(a.i0), _i(a.i1), _i(a.i2),
                                  _i(a.i3), _i(a.i4), _i(a.i5), _i(a.i6))
    elif OP == OP_SSM_M2_SCALE_DACS:
        # d_dacs_state, d_scale, dacs; B, C, H, Q
        mamba2_scale_to_dacs_cell(cell, _q(a.p0), _q(a.p1), _q(a.p2), _i(a.i0), _i(a.i1), _i(a.i2), _i(a.i3))
    elif OP == OP_SSM_M2_DECAY_DACS:
        # d_dacs_decay, d_decay, decay; rows, Q
        mamba2_decay_to_dacs_cell(cell, _q(a.p0), _q(a.p1), _q(a.p2), _i(a.i0), _i(a.i1))
    elif OP == OP_SSM_M2_MERGE_DACS:
        # d_dacs_total, yoff, state, decay; n
        mamba2_merge_dacs_cell(cell, _q(a.p0), _q(a.p1), _q(a.p2), _q(a.p3), _i(a.i0))
    elif OP == OP_SSM_M2_REV_CUMSUM:
        # d_da, d_dacs; B, T, H, C, Q
        mamba2_reverse_cumsum_cell(cell, _q(a.p0), _q(a.p1), _i(a.i0), _i(a.i1), _i(a.i2), _i(a.i3), _i(a.i4))
    elif OP == OP_SSM_M2_YD_MATRIX:
        # d_cb, d_seg, d_y, xd, cb_g, seg_l; B, T, H, C, Q
        mamba2_ydiag_matrix_backward_cell(cell, _q(a.p0), _q(a.p1), _q(a.p2), _q(a.p3), _q(a.p4), _q(a.p5),
                                          _i(a.i0), _i(a.i1), _i(a.i2), _i(a.i3), _i(a.i4))
    elif OP == OP_SSM_M2_CB_BWD:
        # d_b_cb, d_c_cb, d_cb, xbc; B, T, DI, CD, C, Q
        mamba2_cb_backward_cell(cell, _q(a.p0), _q(a.p1), _q(a.p2), _q(a.p3), _i(a.i0), _i(a.i1), _i(a.i2),
                                _i(a.i3), _i(a.i4), _i(a.i5))
    elif OP == OP_SSM_M2_CS_DXD_DB:
        # d_xd_cstate, d_b_cstate, d_b_total, d_b_cb, d_cstate, xd, xbc, decay; B, T, H, DI, CD, C, Q
        mamba2_cstate_dxd_db_cell(cell, _q(a.p0), _q(a.p1), _q(a.p2), _q(a.p3), _q(a.p4), _q(a.p5), _q(a.p6),
                                  _q(a.p7), _i(a.i0), _i(a.i1), _i(a.i2), _i(a.i3), _i(a.i4), _i(a.i5), _i(a.i6))
    elif OP == OP_SSM_M2_SEG_BWD:
        # d_da_seg, d_da_total, d_seg, seg_l, d_da; B, T, H, C, Q
        mamba2_seg_backward_cell(cell, _q(a.p0), _q(a.p1), _q(a.p2), _q(a.p3), _q(a.p4), _i(a.i0), _i(a.i1),
                                 _i(a.i2), _i(a.i3), _i(a.i4))
    elif OP == OP_SSM_M2_DA:
        # d_dt, d_da_total, A; B*T, H
        mamba2_da_cell_body(cell, _q(a.p0), _q(a.p1), _q(a.p2), _i(a.i0), _i(a.i1))
    elif OP == OP_SSM_M2_FOLD_ROWS:
        # part, x, y; rows, cols, rows per tile, tiles, is_dot
        m2_fold_rows_partial_cell(cell, _q(a.p0), _q(a.p1), _q(a.p2), _i(a.i0), _i(a.i1), _i(a.i2), _i(a.i3),
                                  _i(a.i4))
    elif OP == OP_SSM_M2_FOLD_TILES:
        # out_a, out_b, out_s, scale, part; tiles, cols, split, has_scale
        m2_fold_tiles_cell(cell, _q(a.p0), _q(a.p1), _q(a.p2), _q(a.p3), _q(a.p4), _i(a.i0), _i(a.i1),
                           _i(a.i2), _i(a.i3))
    elif OP == OP_SSM_M2_YD_XD:
        # d_xd_ydiag, d_x_from_xd, d_dt_from_xd, d_dt_merged, d_y, cb_g, seg_l, xbc, dt, d_dt, d_xd_cstate,
        # d_xd_total; B, T, H, DI, CD, C, Q
        mamba2_ydiag_xd_backward_cell(cell, _q(a.p0), _q(a.p1), _q(a.p2), _q(a.p3), _q(a.p4), _q(a.p5),
                                      _q(a.p6), _q(a.p7), _q(a.p8), _q(a.p9), _q(a.p10), _q(a.p11), _i(a.i0),
                                      _i(a.i1), _i(a.i2), _i(a.i3), _i(a.i4), _i(a.i5), _i(a.i6))
    elif OP == OP_SSM_M2_MERGE:
        # d_c_total, d_x_total, d_c_yoff, d_c_cb, d_x_from_d, d_x_from_xd; c cells, x cells
        mamba2_postconv_merge_cell(cell, _q(a.p0), _q(a.p1), _q(a.p2), _q(a.p3), _q(a.p4), _q(a.p5), _i(a.i0),
                                   _i(a.i1))
    elif OP == OP_SSM_M2_PACK:
        # p0 the xbc gradient [M, CD]; p1 dx [M, DI], p2 dB [M, N], p3 dC [M, N]; i0 DI, i1 CD (a copy)
        var di = a.i0
        var cd = a.i1
        var row = cell // cd
        var col = cell - row * cd
        if col < di:
            a.p0.unsafe_store(cell, a.p1.unsafe_load(row * di + col))
        elif col < di + M2_D_STATE:
            a.p0.unsafe_store(cell, a.p2.unsafe_load(row * M2_D_STATE + col - di))
        else:
            a.p0.unsafe_store(cell, a.p3.unsafe_load(row * M2_D_STATE + col - di - M2_D_STATE))
