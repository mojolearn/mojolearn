# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The precision Cholesky's operation order (lane fam2-cluster, 2026-10-04).

IDENTICAL, every column: all K components' `L_k L_k^T = cov_k`, `L_k^{-1}`,
`P_k = (L_k^{-1})^T` and `-sum log L_jj` come from ONE launch
(`mstep.mojo::idn_precision_cholesky_kernel`, a block per component) with
ONE readback of the K pivot flags, in place of the per-component chain
(copy, `potrf_lower` reading `info` home once per 32-column panel,
`chol_logdet` reading its scalar home, `set_identity`, `trsm_lower`,
transpose): two drains a component at d <= 32 and about ten at d = 200,
every EM iteration.

THE ORDER, which is a function of d alone and is what the device kernel and
both host columns (`host/gmm_host_oracle.mojo`, `checks/gmm_oracle.mojo`)
compute through `gmm_idn_chol_host` below:

  factor, right-looking, columns j ascending:
      p = a[j][j];  not (p > 0)  ->  info = j + 1, stop
      a[j][j] = ftz(identical_sqrt(p))
      a[i][j] = ftz(identical_div(a[i][j], a[j][j]))                 i > j
      a[r][c] = ftz(identical_mul_add(-a[r][j], a[c][j], a[r][c]))   j < c <= r
  so every cell receives its column updates in ascending j, one pinned fma
  each: the left-looking chain's order, cell by cell, whatever thread runs it.

  inverse, column c, rows i ascending:
      x[c][c] = ftz(identical_div(1, a[c][c]))
      v = 0;  v = ftz(identical_mul_add(-a[i][m], x[m][c], v))  m = c .. i-1
      x[i][c] = ftz(identical_div(v, a[i][i]))                       i > c

  log determinant: s = 0;  s = ftz(s + ftz(identical_log(a[j][j])))  j
  ascending;  log_det_chol = -s.

Every step is one correctly rounded operation (`identical_sqrt`,
`identical_div`, the pinned fma, the portable log) in a fixed order, so the
words are the same on NVIDIA, AMD, Apple and the host. They are NOT the old
chain's words above one panel (the blocked factor's trailing update and the
log determinant's fold both ordered their terms differently): bits move in
this version, on all four columns together. The unblocked factor with a
fused multiply-add per term is at least as accurate as the blocked one.

`-D MOJOLEARN_IDN_GMM_FUSED_CHOL_OFF=1` restores the per-component chain on
the device AND in the host columns (they read this same constant), as does
the master `-D MOJOLEARN_IDN_ALL_OFF=1`. The define must reach the
host-column build as well as the device build.
"""

from std.sys.compile import is_defined

from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_div,
    identical_log,
    identical_mul_add,
    identical_sqrt,
)

#: One thread per column of `L^{-1}`, so d is bounded by the block size.
comptime GMM_IDN_CHOL_TPB = 256
comptime GMM_IDN_CHOL_MAX_D = GMM_IDN_CHOL_TPB

comptime IDN_GMM_FUSED_CHOL = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not (
        # I21 current experiment: NEVER RUN — PENDING VALIDATION; existing defaults preserved.
        is_defined["MOJOLEARN_IDN_GMM_FUSED_CHOL_OFF"]()
        or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
    )
)


def gmm_idn_chol_applies(d: Int) -> Bool:
    """Whether the one-launch order is the fit's order at this d. The device
    driver and both host columns ask this same question."""
    comptime if IDN_GMM_FUSED_CHOL:
        return d <= GMM_IDN_CHOL_MAX_D
    return False


@fieldwise_init
struct GmmIdnChol(Movable):
    """One component's result. `info != 0`: the factor stopped at pivot
    `info - 1` and `l` / `linv` / `logdet` are not meaningful."""

    var info: Int
    var l: List[Float32]
    var linv: List[Float32]
    var logdet: Float32


def gmm_idn_chol_host(
    cov: List[Float32], off: Int, d: Int, jitter: Float32
) -> GmmIdnChol:
    """The module docstring's order on the host, for the `d x d` block of
    `cov` starting at `off`. `l` is the lower factor (zero above the
    diagonal), `linv` its inverse (zero above the diagonal), `logdet` is
    `-sum log L_jj`."""
    var dd = d * d
    var a = List[Float32](capacity=dd)
    for e in range(dd):
        var v = cov[off + e]
        if e // d == e % d:
            v = ftz(ftz(v) + jitter)
        a.append(v)
    var x = List[Float32](length=dd, fill=Float32(0.0))
    for j in range(d):
        var p = a[j * d + j]
        if not (p > Float32(0.0)):
            return GmmIdnChol(j + 1, a^, x^, Float32(0.0))
        var pj = ftz(identical_sqrt(p))
        a[j * d + j] = pj
        for i in range(j + 1, d):
            a[i * d + j] = ftz(identical_div(a[i * d + j], pj))
        for r in range(j + 1, d):
            var arj = a[r * d + j]
            for c in range(j + 1, r + 1):
                a[r * d + c] = ftz(
                    identical_mul_add(-arj, a[c * d + j], a[r * d + c])
                )
    for c in range(d):
        x[c * d + c] = ftz(identical_div(Float32(1.0), a[c * d + c]))
        for i in range(c + 1, d):
            var v = Float32(0.0)
            for m in range(c, i):
                v = ftz(identical_mul_add(-a[i * d + m], x[m * d + c], v))
            x[i * d + c] = ftz(identical_div(v, a[i * d + i]))
    var s = Float32(0.0)
    for j in range(d):
        s = ftz(s + ftz(identical_log(a[j * d + j])))
    for r in range(d):
        for c in range(r + 1, d):
            a[r * d + c] = Float32(0.0)
    return GmmIdnChol(0, a^, x^, -s)
