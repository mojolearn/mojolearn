# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Element functions of the Mamba-3 trapezoid core as a generic tape operation (lane neural-blocks-3-m3,
2026-11-22), dispatched from `autograd/ssm_ops.mojo::apply_ssm` (codes 4860..4899).

Read by both executors through `sequence/dispatch.mojo::apply` (HostExec: a host loop over the elements;
DeviceExec: one GPU thread per element), so the host and the device column run the same function on the
same words. Every element is ONE CALL of a body in `autograd/kernels/mamba3_trapezoid_body.mojo`; no
arithmetic is written in this file.

  op                  elements         body
  OP_SSM_M3_SCAN      batch * H * P    m3t_scan_cell     the forward chain, h at every token, y + D v
  OP_SSM_M3_BWD       batch * H * P    m3t_bwd_cell      the reverse chain: g, v's scan part, the
                                                         coefficient terms per channel
  OP_SSM_M3_DQ        M * H * N        m3t_dq_cell
  OP_SSM_M3_DK        M * H * N        m3t_dk_cell
  OP_SSM_M3_COEF      M * H            m3t_coef_cell     d dt, d A, d trap
  OP_SSM_M3_DV        M * H * P        m3t_dv_cell       v's whole gradient
  OP_SSM_M3_DD        H                m3t_dd_cell       dD
"""
from sequence.ops import Args, FP
from autograd.kernels.mamba3_trapezoid_body import (
    M3P,
    m3t_scan_cell,
    m3t_bwd_cell,
    m3t_dq_cell,
    m3t_dk_cell,
    m3t_coef_cell,
    m3t_dv_cell,
    m3t_dd_cell,
)

comptime OP_SSM_M3_FIRST = 4860
comptime OP_SSM_M3_SCAN = 4860
comptime OP_SSM_M3_BWD = 4861
comptime OP_SSM_M3_DQ = 4862
comptime OP_SSM_M3_DK = 4863
comptime OP_SSM_M3_COEF = 4864
comptime OP_SSM_M3_DV = 4865
comptime OP_SSM_M3_DD = 4866
comptime OP_SSM_M3_LAST = 4899


@always_inline
def _q(p: FP) -> M3P:
    # Args carries executor-owned buffers with an untracked origin; the bodies borrow the same live
    # buffers (no owner moves, no lifetime extension at this boundary).
    return p.unsafe_origin_cast[MutAnyOrigin]()


def apply_ssm_m3[OP: Int](cell: Int, a: Args):
    comptime if OP == OP_SSM_M3_SCAN:
        m3t_scan_cell(cell, _q(a.p0), _q(a.p1), _q(a.p2), _q(a.p3), _q(a.p4), _q(a.p5), _q(a.p6), _q(a.p7),
                      _q(a.p8), a.i0, a.i1, a.i2, a.i3, a.i4)
    elif OP == OP_SSM_M3_BWD:
        # p2 holds the three coefficient-term planes, each M * H * P
        var w = _q(a.p2)
        var n3 = a.i0 * a.i1 * a.i2 * a.i3
        m3t_bwd_cell(cell, _q(a.p0), _q(a.p1), w, w + n3, w + 2 * n3, _q(a.p3), _q(a.p4), _q(a.p5), _q(a.p6),
                     _q(a.p7), _q(a.p8), _q(a.p9), _q(a.p10), a.i0, a.i1, a.i2, a.i3, a.i4)
    elif OP == OP_SSM_M3_DQ:
        m3t_dq_cell(cell, _q(a.p0), _q(a.p1), _q(a.p2), a.i0, a.i2, a.i3, a.i4)
    elif OP == OP_SSM_M3_DK:
        m3t_dk_cell(cell, _q(a.p0), _q(a.p1), _q(a.p2), _q(a.p3), _q(a.p4), _q(a.p5), a.i1, a.i0, a.i2, a.i3,
                    a.i4)
    elif OP == OP_SSM_M3_COEF:
        var w = _q(a.p3)
        var n3 = a.i0 * a.i2 * a.i3
        m3t_coef_cell(cell, _q(a.p0), _q(a.p1), _q(a.p2), w, w + n3, w + 2 * n3, _q(a.p4), _q(a.p5), _q(a.p6),
                      a.i0, a.i2, a.i3)
    elif OP == OP_SSM_M3_DV:
        m3t_dv_cell(cell, _q(a.p0), _q(a.p1), _q(a.p2), _q(a.p3), a.i0, a.i2, a.i3)
    elif OP == OP_SSM_M3_DD:
        m3t_dd_cell(cell, _q(a.p0), _q(a.p1), _q(a.p2), a.i0, a.i2, a.i3)
