# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# SHIPS: compiled into a CPU host binding (python/mojolearn/host_surface.py names which); product, not only a check.
"""Profile `mojolearn.identical.cholesky.fp32.v1` on the HOST: the one-shot
entries of `cholesky/estimator.mojo` restated without a device (workstream
E, the gp host lane, 2026-09-14).

WHAT THIS IS. `cholesky_factor_host` and `cholesky_solve_host` run their
arithmetic in `cholesky/checks/potrf.mojo` and `cholesky/checks/trsm.mojo`
device kernels. This file is a SECOND spelling of those kernels and of the
driver loop, in the driver's order, with no `DeviceContext`, no trace and no
launch geometry, so the `gp` CPU host binding can serve the GP fit and the
Cholesky door on a CPU-only install. It imports only the
`checks/numerics.mojo` seams and `gemm/host/gemm_oracle.mojo` (the gemm
profile's normative answer, GPU-free host code a CPU binding already ships).
`cholesky/checks/cholesky_oracle.mojo::oracle_potrf_lower` is the prior art
the Apple gate holds the device to bit for bit; it records card stages
through `core/identity_trace.mojo` and is therefore not importable into a
host binding, which is the only reason this file restates it.

THE STAGES, EACH WITH THE DEVICE LINE IT MIRRORS

    chol_host_validate_matrix   potrf.mojo:522-610 (chol_validate_matrix),
                                the same relative symmetry test at 2^-20
    chol_host_validate_jitter   potrf.mojo:468-507 (chol_validate_jitter)
    the panel width             potrf.mojo:422-465 (chol_nb_for): 32 under
                                IDENTICAL at every n, even n < 32, and the
                                width is clamped per panel by the driver
    the ridge                   potrf.mojo:618-630 (jitter_diag_kernel):
                                ftz(ftz(A_ii) + jitter)
    the panel                   potrf.mojo:633-697 (panel_factor_kernel):
                                columns ascending, the pivot as
                                not (s > 0) on the ftz chain, the early
                                exit leaving column jc unwritten
    info                        potrf.mojo:1063-1069, info = jc + 1, stop
    the panel solve             trsm.mojo:157-199 (trsm_panel_kernel)
    the trailing update         potrf.mojo:1100-1149: pack L21, the gemm
                                profile's product at OP_NT with k = w, the
                                lower triangle subtracted as ftz(cur - upd)
    the upper triangle          potrf.mojo:759-777 (zero_upper_kernel),
                                +0.0, only when info == 0
    the log-determinant         potrf.mojo:780-818 (logdet_kernel), one
                                ascending chain of identical_log, doubled
                                through identical_mul
    the solve                   trsm.mojo:85-154 (trsm_lower_kernel then
                                trsm_upper_kernel), k ascending in both

DEVIATION 258 (NVIDIA's sqrt one ulp off on some patterns) is why every root
here is `identical_sqrt`, `portable_sqrtf` under IDENTICAL, the same seam the
device panel takes. Every divide is `identical_div` (DEVIATION 1643).

THE TRAILING UPDATE COMPUTES ONLY THE LOWER TRIANGLE. The device multiplies
the whole `n_trail x n_trail` product and subtracts the lower half
(DEVIATION 1636); a cell's value does not depend on which other cells were
computed, so the host evaluates `gemm_oracle_cell` on `j <= i` only. The
upper half of the trailing block is never read by the factorization and is
zeroed at the end, exactly as on the device.

THE SABOTAGE. This file carries no arm of its own. Under
`-D MOJOLEARN_HOST_SABOTAGE=1` the trailing update moves anyway, because
`gemm_oracle_cell` walks its leaf descending
(`gemm/host/gemm_oracle.mojo::GEMM_ORACLE_HOST_SABOTAGE`).
"""

from std.memory import bitcast

from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_div,
    identical_log,
    identical_mul,
    identical_mul_add,
    identical_sqrt,
)
from gemm.host.identical_gemm import OP_NT, contract_leaf_size, gemm_oracle_cell


#: `cholesky/checks/potrf.mojo::CHOL_PROFILE`.
comptime CHOL_HOST_PROFILE = "mojolearn.identical.cholesky.fp32.v1"

#: `potrf.mojo::CHOL_NB_PINNED`. NUMERIC (DEVIATION 1630).
comptime CHOL_HOST_NB_PINNED = 32

#: `potrf.mojo::CHOL_JITTER_BITS`, 2^-20 (DEVIATION 1637).
comptime CHOL_HOST_JITTER_BITS: UInt32 = 0x35800000

#: `potrf.mojo::CHOL_SYM_REL_TOL_BITS`, 2^-20 (DEVIATION 1638).
comptime CHOL_HOST_SYM_REL_TOL_BITS: UInt32 = 0x35800000


@fieldwise_init
struct CholHostFactor(Copyable, Movable):
    """`cholesky/estimator.mojo::CholeskyFactor`, field for field."""

    var l: List[Float32]
    var n: Int
    var info: Int
    var logdet: Float32
    var nb: Int
    var jitter: Float32


def chol_host_jitter_pinned() -> Float32:
    """`potrf.mojo::chol_jitter_pinned`, by its bits."""
    return bitcast[DType.float32](CHOL_HOST_JITTER_BITS)


def chol_host_hex32_bits(v: Float32) -> String:
    """Eight lowercase hex digits of a float32's bit pattern."""
    comptime DIGITS = "0123456789abcdef"
    var u = bitcast[DType.uint32](v)
    var out = String("")
    for i in range(8):
        var nib = Int((u >> UInt32(28 - 4 * i)) & UInt32(0xF))
        out += String(DIGITS[byte=nib])
    return out


def chol_host_nb(n: Int) raises -> Int:
    """`chol_nb_for(n, CHOL_NB_PINNED)`: the pinned width under IDENTICAL,
    unclamped (the driver clamps each panel), and the hint clamped to
    `[1, n]` in any other mode."""
    if n <= 0:
        raise Error("chol_nb_for: n must be positive, got " + String(n))
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return CHOL_HOST_NB_PINNED
    var nb = CHOL_HOST_NB_PINNED
    if nb > n:
        nb = n
    return nb


def chol_host_validate_jitter(jitter: Float32) raises:
    """`potrf.mojo::chol_validate_jitter`, in its order and words."""
    if jitter != jitter:
        raise Error("add_jitter: jitter is NaN; refused by name")
    if jitter < Float32(0.0):
        raise Error(
            "add_jitter: jitter must be non-negative, got a negative value"
            " with bits 0x"
            + chol_host_hex32_bits(jitter)
            + "; a negative ridge subtracts from the diagonal and turns a"
            " positive-definite matrix indefinite"
        )
    var big = bitcast[DType.float32](UInt32(0x7F800000))
    if jitter == big:
        raise Error("add_jitter: jitter is +inf; refused by name")
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        var pinned = chol_host_jitter_pinned()
        var bits = bitcast[DType.uint32](jitter)
        if not (bits == UInt32(0) or bits == bitcast[DType.uint32](pinned)):
            raise Error(
                "add_jitter: NUMERIC_IDENTICAL refuses the unpinned jitter"
                " 0x"
                + chol_host_hex32_bits(jitter)
                + ". The ridge is part of profile "
                + CHOL_HOST_PROFILE
                + " and not a caller's free choice. The two pinned values are"
                " 0x00000000 (no ridge) and 0x"
                + chol_host_hex32_bits(pinned)
                + " (2^-20). DEVIATION 1637. For a larger ridge, apply the"
                " pinned one more than once and record how many times; to"
                " change the value, that is a v2."
            )


def chol_host_validate_matrix(a: List[Float32], n: Int, what: String) raises:
    """`potrf.mojo::chol_validate_matrix`: finite and symmetric within the
    pinned relative tolerance, naming the cell. Host code on both paths."""
    if n <= 0:
        raise Error(
            "cholesky: " + what + " must have a positive dimension, got n="
            + String(n)
        )
    if len(a) != n * n:
        raise Error(
            "cholesky: "
            + what
            + " holds "
            + String(len(a))
            + " floats, an "
            + String(n)
            + " x "
            + String(n)
            + " row-major matrix needs "
            + String(n * n)
        )
    for i in range(n):
        for j in range(n):
            var v = a[i * n + j]
            if v != v:
                raise Error(
                    "cholesky: "
                    + what
                    + " contains NaN at ["
                    + String(i)
                    + ", "
                    + String(j)
                    + "]; refused by name before any launch (IDENTITY_PATHS"
                    " row 39)"
                )
            if v > Float32(3.4028234663852886e38) or v < Float32(
                -3.4028234663852886e38
            ):
                raise Error(
                    "cholesky: "
                    + what
                    + " contains infinity at ["
                    + String(i)
                    + ", "
                    + String(j)
                    + "]; refused by name"
                )
    var tol = bitcast[DType.float32](CHOL_HOST_SYM_REL_TOL_BITS)
    for i in range(n):
        for j in range(i):
            var lo = a[i * n + j]
            var up = a[j * n + i]
            var d = lo - up
            if d < Float32(0.0):
                d = -d
            var m = lo
            if m < Float32(0.0):
                m = -m
            var mu = up
            if mu < Float32(0.0):
                mu = -mu
            if mu > m:
                m = mu
            if d > tol * m:
                raise Error(
                    "cholesky: "
                    + what
                    + " is not symmetric at ["
                    + String(i)
                    + ", "
                    + String(j)
                    + "]: lower 0x"
                    + chol_host_hex32_bits(lo)
                    + " upper 0x"
                    + chol_host_hex32_bits(up)
                    + ", relative difference exceeds CHOL_SYM_REL_TOL"
                    " (2^-20). DEVIATION 1638"
                )


def chol_host_potrf(a_in: List[Float32], n: Int, jitter: Float32) raises -> CholHostFactor:
    """`cholesky/estimator.mojo::cholesky_factor_host`, on the host: validate,
    ridge, factor in `potrf_lower`'s panel order, log-determinant when the
    factorization succeeded. The stage table in this file's header gives
    the device line of every step."""
    chol_host_validate_matrix(a_in, n, String("the matrix"))
    chol_host_validate_jitter(jitter)
    return chol_host_factor_lower(a_in, n, jitter)


def chol_host_factor_lower(
    a_in: List[Float32], n: Int, jitter: Float32
) raises -> CholHostFactor:
    """`jitter_diag_kernel` then `potrf_lower` and the log-determinant, with
    NO host validation: the arithmetic `chol_host_potrf` runs after its two
    refusals. The callers that reach `potrf_lower` without
    `cholesky_factor_host`'s validation on the device (the kernel ridge
    solve, the Gaussian mixture's precision Cholesky, whose covariance is
    not bitwise symmetric) call this, so the host refuses nothing the device
    would have factored. Added for the kernel_methods and mixture host
    families (2026-09-15); `chol_host_potrf`'s bits are unchanged."""
    var nb = chol_host_nb(n)
    var a = a_in.copy()

    # jitter_diag_kernel
    for i in range(n):
        var d = ftz(a[i * n + i])
        a[i * n + i] = ftz(d + jitter)

    var info = 0
    var j0 = 0
    while j0 < n:
        var w = nb
        if j0 + w > n:
            w = n - j0
        var n_trail = n - j0 - w

        # ---- panel_factor_kernel ----------------------------------------
        for c in range(w):
            var jc = j0 + c
            var s = ftz(a[jc * n + jc])
            for k in range(j0, jc):
                var v = ftz(a[jc * n + k])
                s = ftz(identical_mul_add(-v, v, s))
            if not (s > Float32(0.0)):
                # The device's thread 0 stores info and raises the flag; no
                # thread writes column jc.
                info = jc + 1
                break
            a[jc * n + jc] = ftz(identical_sqrt(s))
            var ljj = ftz(a[jc * n + jc])
            for i in range(c + 1, w):
                var r = j0 + i
                var t = ftz(a[r * n + jc])
                for k in range(j0, jc):
                    var lrk = ftz(a[r * n + k])
                    var lck = ftz(a[jc * n + k])
                    t = ftz(identical_mul_add(-lrk, lck, t))
                a[r * n + jc] = ftz(identical_div(t, ljj))
        if info != 0:
            break

        if n_trail > 0:
            # ---- trsm_panel_kernel --------------------------------------
            for idx in range(n_trail):
                var r = j0 + w + idx
                for c in range(w):
                    var jc = j0 + c
                    var t = ftz(a[r * n + jc])
                    for k in range(j0, jc):
                        var lrk = ftz(a[r * n + k])
                        var lck = ftz(a[jc * n + k])
                        t = ftz(identical_mul_add(-lrk, lck, t))
                    var ljj = ftz(a[jc * n + jc])
                    a[r * n + jc] = ftz(identical_div(t, ljj))

            # ---- pack_panel_kernel, identical_gemm_into (OP_NT), the
            # ---- lower subtract ------------------------------------------
            var packed = List[Float32](capacity=n_trail * w)
            for i in range(n_trail):
                for c in range(w):
                    packed.append(a[(j0 + w + i) * n + j0 + c])
            var leaf = contract_leaf_size(w)
            for i in range(n_trail):
                for j in range(i + 1):
                    var upd = ftz(
                        gemm_oracle_cell(
                            packed, packed, OP_NT, i, j, n_trail, n_trail, w, leaf
                        )
                    )
                    var at = (j0 + w + i) * n + j0 + w + j
                    var cur = ftz(a[at])
                    a[at] = ftz(cur - upd)
            _ = packed^

        j0 += nb

    var logdet = Float32(0.0)
    if info == 0:
        # zero_upper_kernel (DEVIATION 1640)
        for i in range(n):
            for j in range(i + 1, n):
                a[i * n + j] = Float32(0.0)
        # copy_vector_from_matrix_diagonal_kernel, then logdet_kernel
        var acc = Float32(0.0)
        for j in range(n):
            acc = ftz(acc + ftz(identical_log(ftz(a[j * n + j]))))
        logdet = ftz(identical_mul(Float32(2.0), acc))
    return CholHostFactor(a^, n, info, logdet, nb, jitter)


def chol_host_trsm_lower(
    l: List[Float32], mut b: List[Float32], n: Int, nrhs: Int
):
    """`trsm_lower_kernel` at `ld == n`, in place over `b` (`n x nrhs`
    row-major). One right-hand side at a time; `k` ascending."""
    for j in range(nrhs):
        for i in range(n):
            var t = ftz(b[i * nrhs + j])
            for k in range(i):
                var lik = ftz(l[i * n + k])
                var bk = ftz(b[k * nrhs + j])
                t = ftz(identical_mul_add(-lik, bk, t))
            var lii = ftz(l[i * n + i])
            b[i * nrhs + j] = ftz(identical_div(t, lii))


def chol_host_trsm_upper(
    l: List[Float32], mut b: List[Float32], n: Int, nrhs: Int
):
    """`trsm_upper_kernel` at `ld == n`: rows descending, the inner sum
    ascending over the rows below."""
    for j in range(nrhs):
        for ii in range(n):
            var i = n - 1 - ii
            var t = ftz(b[i * nrhs + j])
            for k in range(i + 1, n):
                var lki = ftz(l[k * n + i])
                var bk = ftz(b[k * nrhs + j])
                t = ftz(identical_mul_add(-lki, bk, t))
            var lii = ftz(l[i * n + i])
            b[i * nrhs + j] = ftz(identical_div(t, lii))


def chol_host_solve(
    factor: CholHostFactor, b: List[Float32], nrhs: Int
) raises -> List[Float32]:
    """`cholesky/estimator.mojo::cholesky_solve_host`: the same three
    refusals in the same order, then `cho_solve` (forward, then back)."""
    if factor.info != 0:
        raise Error(
            "cholesky_solve_host: refusing to solve against a FAILED"
            " factorization (info="
            + String(factor.info)
            + "). The leading minor of order "
            + String(factor.info)
            + " was not positive definite, so columns "
            + String(factor.info - 1)
            + " onward of the factor are unfinished and solving against"
            " them returns infinities that look like numbers. Add a ridge"
            " (DEVIATION 1637) or fix the matrix. DEVIATION 1634"
        )
    var n = factor.n
    if nrhs <= 0:
        raise Error(
            "cholesky_solve_host: nrhs must be positive, got " + String(nrhs)
        )
    if len(b) != n * nrhs:
        raise Error(
            "cholesky_solve_host: the right-hand side holds "
            + String(len(b))
            + " floats, "
            + String(n)
            + " x "
            + String(nrhs)
            + " needs "
            + String(n * nrhs)
        )
    for i in range(len(b)):
        var v = b[i]
        if v != v:
            raise Error(
                "cholesky_solve_host: the right-hand side contains NaN at"
                " flat index "
                + String(i)
                + "; refused by name (DEVIATION 1638)"
            )
    var x = b.copy()
    chol_host_trsm_lower(factor.l, x, n, nrhs)
    chol_host_trsm_upper(factor.l, x, n, nrhs)
    return x^
