# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Element bodies of the state-space cores as generic tape operations (lane neural-blocks-2, 2026-11-22).

Read by both executors through `sequence/dispatch.mojo::apply` (HostExec: a host loop over the elements;
DeviceExec: one GPU thread per element), so the host and the device column run the same function on the
same words by construction.

Mamba-1 selective scan (`autograd/ssm_primitives.mojo::graph_selective_scan`, the public SelectiveScan
layer): every element here is ONE CALL of the Mamba-1 block kernel's own thread body
(`autograd/kernels/mamba1_scan_body.mojo`): no recurrence is written in this file. The state size is a
compile-time register width in those bodies; this file instantiates them for 4, 8 and 16 states
(`SSM_M1_STATES`) and the graph function refuses any other.

  op                  elements      body
  OP_SSM_M1_A         dim * N       m1_a_from_a_log          A = -exp(A_log)
  OP_SSM_M1_SCAN      batch * dim   m1_scan_cell             S5-S11, y and out = y + u D
  OP_SSM_M1_CKPT      batch * dim   m1_checkpoint_cell       h at every token (backward)
  OP_SSM_M1_BWD       batch * dim   m1_bwd_scan_cell         dh, du (scan part), ddelta, w
  OP_SSM_M1_DBC       M * N         m1_dbc_cell              dB, dC over the gemm contract's leaves
  OP_SSM_M1_DAP       batch * dim   m1_da_partial_acc        per-batch dA partials
  OP_SSM_M1_FOLD      dim * N       m1_param_fold_cell       dA = partials folded in batch order
  OP_SSM_M1_DALOG     dim * N       m1_da_log                dA_log = dA * A
  OP_SSM_M1_DU        M * dim       m1_skip_grads            du = dsk D + du_scan; pD = dsk u (dD's terms)
  OP_SSM_S4D_INIT     dim * N       (init)                   A_log[d, n] = log(n + 1), Mamba's S4D-real init
"""
from sequence.ops import Args, FP
from checks.numerics import ftz, identical_log
from autograd.kernels.mamba1_scan_body import (
    M1P,
    m1_a_from_a_log,
    m1_scan_cell,
    m1_checkpoint_cell,
    m1_bwd_scan_cell,
    m1_dbc_cell,
    m1_da_partial_acc,
    m1_param_fold_cell,
    m1_da_log,
    m1_skip_grads,
)

comptime OP_SSM_FIRST = 4800
comptime OP_SSM_M1_A = 4800
comptime OP_SSM_M1_SCAN = 4801
comptime OP_SSM_M1_CKPT = 4802
comptime OP_SSM_M1_BWD = 4803
comptime OP_SSM_M1_DBC = 4804
comptime OP_SSM_M1_DAP = 4805
comptime OP_SSM_M1_FOLD = 4806
comptime OP_SSM_M1_DALOG = 4807
comptime OP_SSM_M1_DU = 4808
comptime OP_SSM_S4D_INIT = 4809
comptime OP_SSM_LAST = 4819


def ssm_m1_states_supported(n: Int) -> Bool:
    """The state sizes the selective scan's bodies are instantiated for."""
    return n == 4 or n == 8 or n == 16


@always_inline
def _q(p: FP) -> M1P:
    # Args carries executor-owned buffers with an untracked origin; the bodies borrow the same live
    # buffers (no owner moves, no lifetime extension at this boundary).
    return p.unsafe_origin_cast[MutAnyOrigin]()


@always_inline
def _scan[N: Int](cell: Int, a: Args):
    m1_scan_cell[N](cell, _q(a.p0), _q(a.p1), _q(a.p2), _q(a.p3), _q(a.p4), _q(a.p5), _q(a.p6), _q(a.p7),
                    _q(a.p8), a.i0, a.i1, a.i2)


@always_inline
def _ckpt[N: Int](cell: Int, a: Args):
    m1_checkpoint_cell[N](cell, _q(a.p0), _q(a.p1), _q(a.p2), _q(a.p3), _q(a.p4), _q(a.p5), a.i0, a.i1, a.i2)


@always_inline
def _bwd[N: Int](cell: Int, a: Args):
    m1_bwd_scan_cell[N](cell, _q(a.p0), _q(a.p1), _q(a.p2), _q(a.p3), _q(a.p4), _q(a.p5), _q(a.p6), _q(a.p7),
                        _q(a.p8), _q(a.p9), _q(a.p10), a.i0, a.i1, a.i2)


@always_inline
def _dbc[N: Int](cell: Int, a: Args):
    m1_dbc_cell[N](cell, _q(a.p0), _q(a.p1), _q(a.p2), _q(a.p3), _q(a.p4), _q(a.p5), a.i0 * a.i1, a.i1, a.i2,
                   a.i4, a.i5)


@always_inline
def _dap[N: Int](cell: Int, a: Args):
    var batch = a.i0
    var dim = a.i2
    if cell >= batch * dim:
        return
    var bb = cell // dim
    var d = cell - bb * dim
    var acc = m1_da_partial_acc[N](bb, d, _q(a.p1), _q(a.p2), _q(a.p3), _q(a.p4), _q(a.p5), _q(a.p6), a.i1, dim)
    comptime for n in range(N):
        a.p0.unsafe_store((bb * dim + d) * N + n, acc[n])


@always_inline
def _by_states[OP: Int](cell: Int, a: Args):
    # a.i3 is the state size, checked by the graph function (ssm_m1_states_supported)
    var n = a.i3
    comptime if OP == OP_SSM_M1_SCAN:
        if n == 16:
            _scan[16](cell, a)
        elif n == 8:
            _scan[8](cell, a)
        else:
            _scan[4](cell, a)
    elif OP == OP_SSM_M1_CKPT:
        if n == 16:
            _ckpt[16](cell, a)
        elif n == 8:
            _ckpt[8](cell, a)
        else:
            _ckpt[4](cell, a)
    elif OP == OP_SSM_M1_BWD:
        if n == 16:
            _bwd[16](cell, a)
        elif n == 8:
            _bwd[8](cell, a)
        else:
            _bwd[4](cell, a)
    elif OP == OP_SSM_M1_DBC:
        if n == 16:
            _dbc[16](cell, a)
        elif n == 8:
            _dbc[8](cell, a)
        else:
            _dbc[4](cell, a)
    elif OP == OP_SSM_M1_DAP:
        if n == 16:
            _dap[16](cell, a)
        elif n == 8:
            _dap[8](cell, a)
        else:
            _dap[4](cell, a)


def apply_ssm[OP: Int](cell: Int, a: Args):
    comptime if OP == OP_SSM_M1_A:
        a.p1.unsafe_store(cell, m1_a_from_a_log(a.p0.unsafe_load(cell)))
    elif OP == OP_SSM_M1_SCAN or OP == OP_SSM_M1_CKPT or OP == OP_SSM_M1_BWD or OP == OP_SSM_M1_DBC or OP == OP_SSM_M1_DAP:
        _by_states[OP](cell, a)
    elif OP == OP_SSM_M1_FOLD:
        m1_param_fold_cell(cell, _q(a.p0), _q(a.p1), a.i0, a.i1)
    elif OP == OP_SSM_M1_DALOG:
        a.p0.unsafe_store(cell, m1_da_log(a.p1.unsafe_load(cell), a.p2.unsafe_load(cell)))
    elif OP == OP_SSM_M1_DU:
        # p0 dsk (the upstream of out), p1 D [dim], p2 u, p3 du_scan; writes p4 du, p5 pD. The join is the
        # block's (mamba_bwd_du_join_kernel): the D-skip term first, then the scan term; a third consumer of
        # u (Mamba's x_proj) is added by the tape afterwards, the block's order.
        var d = cell % a.i0
        var g = m1_skip_grads(ftz(a.p0.unsafe_load(cell)), ftz(a.p1.unsafe_load(d)), ftz(a.p2.unsafe_load(cell)))
        a.p4.unsafe_store(cell, ftz(ftz(g[0]) + ftz(a.p3.unsafe_load(cell))))
        a.p5.unsafe_store(cell, g[1])
    elif OP == OP_SSM_S4D_INIT:
        a.p0.unsafe_store(cell, ftz(identical_log(Float32(cell % a.i0 + 1))))
