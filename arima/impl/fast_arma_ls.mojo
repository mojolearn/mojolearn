"""FAST-only `_arma_least_squares` with one threadgroup per series (Apple).

`arma_least_squares_kernel` is one thread per series, serial in the series
length: at 200k observations the start-parameter solve runs a Householder QR
over a 200k-row design in ONE GPU thread, with every matrix materialized in
scratch. Here a threadgroup owns a series and never builds a matrix: each
thread reads its rows straight from `y` (the lagged columns are shifted
reads, the MA columns are the AR pre-fit residual recomputed from `y`) and
folds them into a private `R` and `Q'b` with Givens rotations; thread 0
merges the per-thread `R`s the same way and back-substitutes. Orthogonal
updates keep the conditioning of the QR, not of the normal equations.

Same nine steps as the reference, same rank test (`|R_jj| <= LS_RANK_TOL *
max |R_jj|`, and the diagonal magnitudes of a QR are unique), same
degenerate arm, same `test_invparams` verdict. FAST arithmetic: the row
order differs, so the values agree to rounding, not bitwise.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.math import sqrt
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from arima.impl.linalg.batched.least_squares import LS_RANK_TOL

comptime FLS_TPB = 64
comptime FLS_MAX_COLS = 8
comptime FLS_RSZ = FLS_MAX_COLS * FLS_MAX_COLS + FLS_MAX_COLS


@always_inline
def _givens_row(
    mut r: InlineArray[Float32, FLS_MAX_COLS * FLS_MAX_COLS],
    mut qb: InlineArray[Float32, FLS_MAX_COLS],
    mut a: InlineArray[Float32, FLS_MAX_COLS],
    b_in: Float32,
    n: Int,
):
    """Fold the row `(a, b)` into the upper-triangular `R` and `Q'b`."""
    var b = b_in
    comptime for j in range(FLS_MAX_COLS):
        if j < n:
            var aj = a[j]
            if aj != Float32(0.0):
                var rjj = r[j * FLS_MAX_COLS + j]
                var h = sqrt(rjj * rjj + aj * aj)
                var c = rjj / h
                var s = aj / h
                r[j * FLS_MAX_COLS + j] = h
                comptime for l in range(j + 1, FLS_MAX_COLS):
                    if l < n:
                        var t = r[j * FLS_MAX_COLS + l]
                        r[j * FLS_MAX_COLS + l] = c * t + s * a[l]
                        a[l] = c * a[l] - s * t
                var tb = qb[j]
                qb[j] = c * tb + s * b
                b = c * b - s * tb


def _fls_solve[so: MutOrigin, xo: MutOrigin, io: MutOrigin](
    sh: MutPointer[Float32, so, address_space = AddressSpace.SHARED],
    tid: Int,
    n: Int,
    mut r: InlineArray[Float32, FLS_MAX_COLS * FLS_MAX_COLS],
    mut qb: InlineArray[Float32, FLS_MAX_COLS],
    x_out: MutPointer[Float32, xo, address_space = AddressSpace.SHARED],
    info_out: MutPointer[Int32, io, address_space = AddressSpace.SHARED],
):
    """Every thread publishes its `R`, thread 0 merges them, checks the
    rank and back-substitutes into `x_out`; `info_out[0]` is 0 or `j + 1`
    for the first rank-deficient column. Ends on a barrier."""
    comptime for t in range(FLS_MAX_COLS * FLS_MAX_COLS):
        sh[tid * FLS_RSZ + t] = r[t]
    comptime for t in range(FLS_MAX_COLS):
        sh[tid * FLS_RSZ + FLS_MAX_COLS * FLS_MAX_COLS + t] = qb[t]
    barrier()
    if tid == 0:
        var a = InlineArray[Float32, FLS_MAX_COLS](fill=0.0)
        for o in range(1, FLS_TPB):
            for row in range(n):
                comptime for c in range(FLS_MAX_COLS):
                    a[c] = sh[o * FLS_RSZ + row * FLS_MAX_COLS + c]
                _givens_row(
                    r, qb, a,
                    sh[o * FLS_RSZ + FLS_MAX_COLS * FLS_MAX_COLS + row], n,
                )
        var rmax = Float32(0.0)
        for j in range(n):
            rmax = max(rmax, abs(r[j * FLS_MAX_COLS + j]))
        var info = Int32(0)
        if rmax == Float32(0.0):
            info = Int32(1)
        else:
            for j in range(n):
                if info == 0 and abs(r[j * FLS_MAX_COLS + j]) <= LS_RANK_TOL * rmax:
                    info = Int32(j + 1)
        if info == 0:
            var j = n - 1
            while j >= 0:
                var acc = qb[j]
                for l in range(j + 1, n):
                    acc -= r[j * FLS_MAX_COLS + l] * x_out[l]
                x_out[j] = acc / r[j * FLS_MAX_COLS + j]
                j -= 1
        info_out[0] = info
    barrier()


def fast_arma_ls_kernel(
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
    """Steps 1 to 7 of `arma_least_squares_kernel` for series
    `block_idx.x`. Writes `info` (negative for the pre-fit) and, only when
    it is 0, the parameters; the degenerate fill and `test_invparams` run
    after, in `estimate_x0.mojo::fast_ls_finish_kernel`."""
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
        FLS_TPB * FLS_RSZ, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var arfit = stack_allocation[
        FLS_MAX_COLS, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var afit = stack_allocation[
        FLS_MAX_COLS, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var sinfo = stack_allocation[
        1, Scalar[DType.int32], address_space=AddressSpace.SHARED
    ]()
    var r = InlineArray[Float32, FLS_MAX_COLS * FLS_MAX_COLS](fill=0.0)
    var qb = InlineArray[Float32, FLS_MAX_COLS](fill=0.0)
    var a = InlineArray[Float32, FLS_MAX_COLS](fill=0.0)

    # -- 1. the AR(p_ar) pre-fit whose residual stands in for the MA lags
    if q != 0:
        var m2 = n_obs_d - p_ar
        var i = tid
        while i < m2:
            comptime for c in range(FLS_MAX_COLS):
                a[c] = y[yb + p_ar - c - 1 + i] if c < p_ar else Float32(0.0)
            _givens_row(r, qb, a, y[yb + p_ar + i], p_ar)
            i += FLS_TPB
        _fls_solve(sh, tid, p_ar, r, qb, arfit, sinfo)
        if sinfo[0] != 0:
            if tid == 0:
                info[bid] = -sinfo[0]
            return
        comptime for t in range(FLS_MAX_COLS * FLS_MAX_COLS):
            r[t] = 0.0
        comptime for t in range(FLS_MAX_COLS):
            qb[t] = 0.0

    # -- 2 to 5. the ARMA design, row by row, and its fit
    var ar_offset = r_ls - p * s
    var res_offset = r_ls - p_ar - q * s
    var i = tid
    while i < m1:
        comptime for c in range(FLS_MAX_COLS):
            a[c] = 0.0
        if k != 0:
            a[0] = 1.0
        for lag in range(p):
            a[k + lag] = y[yb + ar_offset + s * (p - lag - 1) + i]
        for lag in range(q):
            # resid[j] = y[p_ar + j] - sum_c y[p_ar - c - 1 + j] * arfit[c]
            var j = res_offset + s * (q - lag - 1) + i
            var acc = y[yb + p_ar + j]
            for c in range(p_ar):
                acc -= y[yb + p_ar - c - 1 + j] * arfit[c]
            a[k + p + lag] = acc
        _givens_row(r, qb, a, y[yb + r_ls + i], ncols)
        i += FLS_TPB
    _fls_solve(sh, tid, ncols, r, qb, afit, sinfo)
    if sinfo[0] != 0:
        if tid == 0:
            info[bid] = sinfo[0]
        return

    # -- 7. sigma2 over rows q .. m1 of the final residual
    if est_sigma2_in != 0:
        var part = Float32(0.0)
        i = q + tid
        while i < m1:
            var res = y[yb + r_ls + i]
            if k != 0:
                res -= afit[0]
            for lag in range(p):
                res -= y[yb + ar_offset + s * (p - lag - 1) + i] * afit[k + lag]
            for lag in range(q):
                var j = res_offset + s * (q - lag - 1) + i
                var acc = y[yb + p_ar + j]
                for c in range(p_ar):
                    acc -= y[yb + p_ar - c - 1 + j] * arfit[c]
                res -= acc * afit[k + p + lag]
            part += res * res
            i += FLS_TPB
        sh[tid] = part
        barrier()
        var w = FLS_TPB // 2
        while w > 0:
            if tid < w:
                sh[tid] += sh[tid + w]
            barrier()
            w //= 2
        if tid == 0:
            d_sigma2[bid] = sh[0] / Float32(m1 - q)

    # -- 6. the solution into the parameter vectors
    if tid == 0:
        if k != 0:
            d_mu[bid] = afit[0]
        for c in range(p):
            d_ar[p * bid + c] = afit[k + c]
        for c in range(q):
            d_ma[q * bid + c] = afit[k + p + c]
        info[bid] = 0
