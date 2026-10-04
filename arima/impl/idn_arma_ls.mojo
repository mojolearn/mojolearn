# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/fam2-timeseries (2026-10-04): `_arma_least_squares` with one block of
ILS_TPB threads per series under IDENTICAL (candidate arm
`-D MOJOLEARN_IDN_X0_PAR_LS=1`; `arima/impl/idn_ls_math.mojo` has the
arithmetic, the fold order and why the bits are the same on every vendor and
in the host column). The kernel's shape is `fast_arma_ls.mojo`'s: no design
matrix is built, each thread reads its rows straight from `y`."""

from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from arima.impl.idn_ls_math import (
    ILS_MAX_COLS, ILS_R2, ILS_RSZ, ILS_TPB, ils_div, ils_fma, ils_givens_row, ils_solve,
)
from checks.numerics import ftz


@always_inline
def _ils_merge_solve[so: MutOrigin, xo: MutOrigin, io: MutOrigin](
    sh: MutPointer[Float32, so, address_space = AddressSpace.SHARED],
    tid: Int,
    n: Int,
    mut r: InlineArray[Float32, ILS_R2],
    mut qb: InlineArray[Float32, ILS_MAX_COLS],
    x_out: MutPointer[Float32, xo, address_space = AddressSpace.SHARED],
    info_out: MutPointer[Int32, io, address_space = AddressSpace.SHARED],
):
    """Every thread publishes its triangle; round d of the pairwise tree
    has thread t with t % 2^(d+1) == 0 fold the n rows of thread t + 2^d
    (ascending) into its own; thread 0 solves into `x_out` and writes
    `info_out[0]`. Ends on a barrier."""
    for t in range(ILS_R2):
        sh[tid * ILS_RSZ + t] = r[t]
    for t in range(ILS_MAX_COLS):
        sh[tid * ILS_RSZ + ILS_R2 + t] = qb[t]
    barrier()
    var a = InlineArray[Float32, ILS_MAX_COLS](fill=Float32(0.0))
    var stride = 1
    while stride < ILS_TPB:
        if tid % (2 * stride) == 0:
            var o = tid + stride
            for row in range(n):
                for c in range(ILS_MAX_COLS):
                    a[c] = sh[o * ILS_RSZ + row * ILS_MAX_COLS + c]
                ils_givens_row(r, qb, a, sh[o * ILS_RSZ + ILS_R2 + row], n)
            for t in range(ILS_R2):
                sh[tid * ILS_RSZ + t] = r[t]
            for t in range(ILS_MAX_COLS):
                sh[tid * ILS_RSZ + ILS_R2 + t] = qb[t]
        barrier()
        stride *= 2
    if tid == 0:
        var x = InlineArray[Float32, ILS_MAX_COLS](fill=Float32(0.0))
        var info = ils_solve(r, qb, n, x)
        # the at most ILS_MAX_COLS unknowns (zeros past n)
        comptime for j in range(ILS_MAX_COLS):
            x_out[j] = x[j]
        info_out[0] = info
    barrier()


def idn_arma_ls_kernel(
    y: MutPointer[Float32, MutAnyOrigin],
    d_ar: MutPointer[Float32, MutAnyOrigin],
    d_ma: MutPointer[Float32, MutAnyOrigin],
    d_sigma2: MutPointer[Float32, MutAnyOrigin],
    d_mu: MutPointer[Float32, MutAnyOrigin],
    info: MutPointer[Int32, MutAnyOrigin],
    n_obs_d_in: Int32,
    p_in: Int32,
    q_in: Int32,
    s_in: Int32,
    k_in: Int32,
    p_ar_in: Int32,
    r_ls_in: Int32,
    est_sigma2_in: Int32,
):
    """Steps 1 to 7 of `arma_least_squares_kernel` for series `block_idx.x`
    over a block of ILS_TPB threads. Writes `info` (negative for the
    pre-fit) and, only when it is 0, the parameters; the refusal fill and
    `test_invparams` run after, in `estimate_x0.mojo::fast_ls_finish_kernel`.
    The host column: `arima_oracle._arma_least_squares_par`."""
    var bid = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var n_obs_d = Int(n_obs_d_in)
    var p = Int(p_in)
    var q = Int(q_in)
    var s = Int(s_in)
    var k = Int(k_in)
    var p_ar = Int(p_ar_in)
    var r_ls = Int(r_ls_in)
    var m1 = n_obs_d - r_ls
    var ncols = p + q + k
    var yb = bid * n_obs_d
    var sh = stack_allocation[
        ILS_TPB * ILS_RSZ, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var arfit = stack_allocation[
        ILS_MAX_COLS, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var afit = stack_allocation[
        ILS_MAX_COLS, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var sinfo = stack_allocation[
        1, Scalar[DType.int32], address_space=AddressSpace.SHARED
    ]()
    var r = InlineArray[Float32, ILS_R2](fill=Float32(0.0))
    var qb = InlineArray[Float32, ILS_MAX_COLS](fill=Float32(0.0))
    var a = InlineArray[Float32, ILS_MAX_COLS](fill=Float32(0.0))

    # -- 1. the AR(p_ar) pre-fit whose residual stands in for the MA lags
    if q != 0:
        var m2 = n_obs_d - p_ar
        var i = tid
        while i < m2:
            for c in range(ILS_MAX_COLS):
                a[c] = ftz(y[yb + p_ar - c - 1 + i]) if c < p_ar else Float32(0.0)
            ils_givens_row(r, qb, a, ftz(y[yb + p_ar + i]), p_ar)
            i += ILS_TPB
        _ils_merge_solve(sh, tid, p_ar, r, qb, arfit, sinfo)
        if sinfo[0] != 0:
            if tid == 0:
                info[bid] = -sinfo[0]
            return
        for t in range(ILS_R2):
            r[t] = Float32(0.0)
        for t in range(ILS_MAX_COLS):
            qb[t] = Float32(0.0)

    # -- 2 to 5. the ARMA design, row by row, and its fit
    var ar_offset = r_ls - p * s
    var res_offset = r_ls - p_ar - q * s
    var i = tid
    while i < m1:
        for c in range(ILS_MAX_COLS):
            a[c] = Float32(0.0)
        if k != 0:
            a[0] = Float32(1.0)
        for lag in range(p):
            a[k + lag] = ftz(y[yb + ar_offset + s * (p - lag - 1) + i])
        for lag in range(q):
            # resid[j] = y[p_ar + j] - sum_c y[p_ar - c - 1 + j] * arfit[c]
            var j = res_offset + s * (q - lag - 1) + i
            var acc = ftz(y[yb + p_ar + j])
            for c in range(p_ar):
                acc = ils_fma(-ftz(y[yb + p_ar - c - 1 + j]), arfit[c], acc)
            a[k + p + lag] = acc
        ils_givens_row(r, qb, a, ftz(y[yb + r_ls + i]), ncols)
        i += ILS_TPB
    _ils_merge_solve(sh, tid, ncols, r, qb, afit, sinfo)
    if sinfo[0] != 0:
        if tid == 0:
            info[bid] = sinfo[0]
        return

    # -- 7. sigma2 over rows q .. m1 of the final residual
    if est_sigma2_in != 0:
        var part = Float32(0.0)
        i = q + tid
        while i < m1:
            var res = ftz(y[yb + r_ls + i])
            if k != 0:
                res = ftz(res - afit[0])
            for lag in range(p):
                res = ils_fma(-ftz(y[yb + ar_offset + s * (p - lag - 1) + i]), afit[k + lag], res)
            for lag in range(q):
                var j = res_offset + s * (q - lag - 1) + i
                var acc = ftz(y[yb + p_ar + j])
                for c in range(p_ar):
                    acc = ils_fma(-ftz(y[yb + p_ar - c - 1 + j]), arfit[c], acc)
                res = ils_fma(-acc, afit[k + p + lag], res)
            part = ils_fma(res, res, part)
            i += ILS_TPB
        sh[tid] = part
        barrier()
        var w = ILS_TPB // 2
        while w > 0:
            if tid < w:
                sh[tid] = ftz(sh[tid] + sh[tid + w])
            barrier()
            w //= 2
        if tid == 0:
            d_sigma2[bid] = ils_div(sh[0], Float32(m1 - q))

    # -- 6. the solution into the parameter vectors
    if tid == 0:
        if k != 0:
            d_mu[bid] = afit[0]
        for c in range(p):
            d_ar[p * bid + c] = afit[k + c]
        for c in range(q):
            d_ma[q * bid + c] = afit[k + p + c]
        info[bid] = 0
