# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/fam2-timeseries (2026-10-04): the arithmetic of the IDENTICAL
parallel start-parameter least squares, shared word for word by the device
kernel (`arima/impl/idn_arma_ls.mojo`) and the host column
(`arima/host/arima_oracle.mojo::_arma_least_squares_par`). No device import.

`arma_least_squares_kernel` is one thread per series, serial in the series
length (a Householder QR over an n_obs-row design in one GPU thread). Here a
block of ILS_TPB threads owns a series: thread `tid` folds rows tid,
tid + ILS_TPB, ... into its own upper-triangular `R` and `Q'b` by Givens
rotations, the per-thread triangles merge in a fixed pairwise tree (round d:
thread t with t % 2^(d+1) == 0 folds thread t + 2^d's rows), thread 0 tests
the rank and back-substitutes. `fast_arma_ls.mojo` is the same shape in FAST
arithmetic; this one pins every operation (one rounding each: `identical_mul`,
`identical_mul_add`, `identical_div`, `identical_sqrt`, flush after each), so
NVIDIA, AMD, Apple and the host column, which replays the threads and the
tree in order, produce the same bits.

BITS CHANGE against the Householder form (a different, equally stable
orthogonal factorization: the starting point moves by rounding, the rank test
is the same `|R_jj| <= LS_RANK_TOL max |R_jj|`).

CANDIDATE ARM, default OFF: `-D MOJOLEARN_IDN_X0_PAR_LS=1` (IDENTICAL only,
off under MOJOLEARN_IDN_ALL_OFF) takes it for series of at least
X0_IDN_PAR_LS_MIN_OBS differenced observations (2048;
`-D MOJOLEARN_IDN_X0_PAR_LS_MIN256=1` lowers the threshold to 256, the second
arm to time), when the systems have at most ILS_MAX_COLS columns."""
from std.sys.compile import is_defined

from checks.numerics import (
    GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_div, identical_mul,
    identical_mul_add, identical_sqrt,
)

comptime X0_IDN_PAR_LS = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and is_defined["MOJOLEARN_IDN_X0_PAR_LS"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
comptime X0_IDN_PAR_LS_MIN_OBS = 256 if is_defined["MOJOLEARN_IDN_X0_PAR_LS_MIN256"]() else 2048

comptime ILS_TPB = 64
comptime ILS_MAX_COLS = 8
comptime ILS_R2 = ILS_MAX_COLS * ILS_MAX_COLS
comptime ILS_RSZ = ILS_R2 + ILS_MAX_COLS
#: `LS_RANK_TOL` (`arima/impl/linalg/batched/least_squares.mojo`)
comptime ILS_RANK_TOL = Float32(1.0e-5)


@always_inline
def ils_mul(a: Float32, b: Float32) -> Float32:
    return ftz(identical_mul(a, b))


@always_inline
def ils_fma(a: Float32, b: Float32, c: Float32) -> Float32:
    return ftz(identical_mul_add(a, b, c))


@always_inline
def ils_div(a: Float32, b: Float32) -> Float32:
    return ftz(identical_div(a, b))


@always_inline
def ils_givens_row(
    mut r: InlineArray[Float32, ILS_R2],
    mut qb: InlineArray[Float32, ILS_MAX_COLS],
    mut a: InlineArray[Float32, ILS_MAX_COLS],
    b_in: Float32,
    n: Int,
):
    """Fold the row `(a, b)` into the upper-triangular `R` (row major,
    ILS_MAX_COLS wide) and `Q'b`: for each column j with a_j != 0, the
    rotation h = sqrt(r_jj^2 + a_j^2), c = r_jj / h, s = a_j / h. `a` is
    consumed. A rotation whose h flushed to zero is skipped."""
    var b = b_in
    for j in range(n):
        var aj = a[j]
        if aj != Float32(0.0):
            var rjj = r[j * ILS_MAX_COLS + j]
            var h = ftz(identical_sqrt(ils_fma(aj, aj, ils_mul(rjj, rjj))))
            if h != Float32(0.0):
                var c = ils_div(rjj, h)
                var s = ils_div(aj, h)
                r[j * ILS_MAX_COLS + j] = h
                for l in range(j + 1, n):
                    var t = r[j * ILS_MAX_COLS + l]
                    var al = a[l]
                    r[j * ILS_MAX_COLS + l] = ils_fma(s, al, ils_mul(c, t))
                    a[l] = ils_fma(-s, t, ils_mul(c, al))
                var tb = qb[j]
                qb[j] = ils_fma(s, b, ils_mul(c, tb))
                b = ils_fma(-s, tb, ils_mul(c, b))


@always_inline
def ils_solve(
    r: InlineArray[Float32, ILS_R2],
    qb: InlineArray[Float32, ILS_MAX_COLS],
    n: Int,
    mut x: InlineArray[Float32, ILS_MAX_COLS],
) -> Int32:
    """The rank test and the back-substitution of the merged triangle:
    returns 0 with the solution in `x`, or j + 1 for the first column whose
    `|R_jj| <= ILS_RANK_TOL max |R_jj|` (1 when the maximum is 0)."""
    var rmax = Float32(0.0)
    for j in range(n):
        var v = abs(r[j * ILS_MAX_COLS + j])
        if v > rmax:
            rmax = v
    if rmax == Float32(0.0):
        return Int32(1)
    var tol = ils_mul(ILS_RANK_TOL, rmax)
    for j in range(n):
        if abs(r[j * ILS_MAX_COLS + j]) <= tol:
            return Int32(j + 1)
    for jj in range(n):
        var j = n - 1 - jj
        var acc = qb[j]
        for l in range(j + 1, n):
            acc = ils_fma(-r[j * ILS_MAX_COLS + l], x[l], acc)
        x[j] = ils_div(acc, r[j * ILS_MAX_COLS + j])
    return Int32(0)
