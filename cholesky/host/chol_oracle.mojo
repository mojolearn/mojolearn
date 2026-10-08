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
from cholesky.logdet_fold import logdet_serial

from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_div,
    identical_log,
    identical_mul,
    identical_mul_add,
    identical_mul_add_simd,
    identical_sqrt,
)
from std.sys.compile import is_defined
from gemm.host.identical_gemm import OP_NT, contract_leaf_size, gemm_oracle_cell
from gemm.host.gemm_oracle import (
    GEMM_ORACLE_SABOTAGE_ORDER_ARM,
    GEMM_ORACLE_SABOTAGE_VALUE_ARM,
    leaf_begin,
    leaf_count,
    leaf_end,
)
from core.host_simd_identical import ftz_v


#: `cholesky/checks/potrf.mojo::CHOL_PROFILE`.
comptime CHOL_HOST_PROFILE = "mojolearn.identical.cholesky.fp32.v1"

#: `potrf.mojo::CHOL_NB_PINNED`. NUMERIC (DEVIATION 1630). lane gap-linalg
#: (2026-10-08): 128 under `-D MOJOLEARN_IDN_CHOL_NB128` (the device's
#: `potrf_blocked.CHOL_IDN_NB128`: the host column moves with NVIDIA and
#: AMD; the loop below is generic in the width, `contract_leaf_size(128)`
#: is still one leaf).
comptime CHOL_HOST_NB_PINNED = 128 if (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and is_defined["MOJOLEARN_IDN_CHOL_NB128"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
) else 32

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


#: THE TRAILING UPDATE, CHOL_W CELLS AT ONCE (lane neighbors-cpu,
#: 2026-09-28). Lane `l` of a register is cell (i, j0 + l) of the lower
#: triangle: `gemm_oracle_cell(packed, packed, OP_NT, i, j, ...)` spelled
#: lane-wise (the same contract leaves, each `ftz(fma(ftz(P[i, c]),
#: ftz(P[j, c]), acc))` ascending from `+0.0`, the leaf seam, the oracle's
#: sabotage arms, the balanced fold), then `ftz(ftz(cur) - ftz(upd))`. The
#: j operands come from a flushed feature-major copy of the panel. Cells
#: above the diagonal of a straddling block are computed and discarded.
#: `-D MOJOLEARN_CHOL_HOST_SCALAR` builds the per-cell loops instead (here
#: and in the two triangular solves; the before arm of the speed record).
comptime CHOL_HOST_SCALAR = is_defined["MOJOLEARN_CHOL_HOST_SCALAR"]()
comptime CHOL_W = 8
comptime CholV = SIMD[DType.float32, CHOL_W]
comptime CholU = SIMD[DType.uint32, CHOL_W]
comptime CholPtr = UnsafePointer[Float32, MutUntrackedOrigin]


def _chol_trailing_lower_v(
    mut a: List[Float32], n: Int, base: Int, packed: List[Float32],
    n_trail: Int, w: Int, leaf: Int,
):
    var pcount = leaf_count(w, leaf)
    var nb = (n_trail + CHOL_W - 1) // CHOL_W
    var pt = List[Float32](length=nb * w * CHOL_W, fill=Float32(0.0))
    var ptp = rebind[CholPtr](pt.unsafe_ptr())
    var pk = rebind[CholPtr](packed.unsafe_ptr())
    for j in range(n_trail):
        var jb = j // CHOL_W
        var l = j % CHOL_W
        for c in range(w):
            ptp.unsafe_store((jb * w + c) * CHOL_W + l, ftz(pk.unsafe_load(j * w + c)))
    var parts = List[Float32](length=(pcount + 1) * CHOL_W, fill=Float32(0.0))
    var sp = rebind[CholPtr](parts.unsafe_ptr())
    var ap = rebind[CholPtr](a.unsafe_ptr())
    for i in range(n_trail):
        var arow = pk + i * w
        var jb_last = i // CHOL_W
        for jb in range(jb_last + 1):
            var panel = ptp + jb * w * CHOL_W
            for t in range(pcount):
                var lo = leaf_begin(t, leaf)
                var hi = leaf_end(t, leaf, w)
                var acc = CholV(0.0)
                for q in range(hi - lo):
                    var c = lo + q
                    comptime if GEMM_ORACLE_SABOTAGE_ORDER_ARM:
                        c = hi - 1 - q
                    acc = ftz_v[CHOL_W](identical_mul_add_simd[CHOL_W](
                        CholV(ftz(arow.unsafe_load(c))),
                        panel.unsafe_load[width=CHOL_W](c * CHOL_W), acc,
                    ))
                acc = ftz_v[CHOL_W](acc)
                comptime if GEMM_ORACLE_SABOTAGE_VALUE_ARM:
                    var b = bitcast[DType.uint32, CHOL_W](acc)
                    var small = (b & CholU(0x7FFFFFFF)).lt(CholU(0x00800000))
                    acc = bitcast[DType.float32, CHOL_W](small.select(CholU(0x00800000), b + CholU(1)))
                sp.unsafe_store(t * CHOL_W, acc)
            var upd: CholV
            if pcount == 1:
                upd = ftz_v[CHOL_W](sp.unsafe_load[width=CHOL_W](0))
            else:
                var width = pcount
                while width > 1:
                    var pairs = width // 2
                    for q in range(pairs):
                        var x = ftz_v[CHOL_W](sp.unsafe_load[width=CHOL_W](2 * q * CHOL_W))
                        var y = ftz_v[CHOL_W](sp.unsafe_load[width=CHOL_W]((2 * q + 1) * CHOL_W))
                        sp.unsafe_store(q * CHOL_W, ftz_v[CHOL_W](x + y))
                    if width % 2 != 0:
                        sp.unsafe_store(pairs * CHOL_W, sp.unsafe_load[width=CHOL_W]((width - 1) * CHOL_W))
                        width = pairs + 1
                    else:
                        width = pairs
                upd = ftz_v[CHOL_W](sp.unsafe_load[width=CHOL_W](0))
            upd = ftz_v[CHOL_W](upd)
            var j0 = jb * CHOL_W
            var at = (base + i) * n + base + j0
            if j0 + CHOL_W <= i + 1:
                var cur = ftz_v[CHOL_W](ap.unsafe_load[width=CHOL_W](at))
                ap.unsafe_store(at, ftz_v[CHOL_W](cur - upd))
            else:
                for l in range(i + 1 - j0):
                    var cur = ftz(ap.unsafe_load(at + l))
                    ap.unsafe_store(at + l, ftz(cur - upd[l]))
    _ = pt^
    _ = parts^


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
            comptime if CHOL_HOST_SCALAR:
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
            else:
                _chol_trailing_lower_v(a, n, j0 + w, packed, n_trail, w, leaf)
            _ = packed^

        j0 += nb

    var logdet = Float32(0.0)
    if info == 0:
        # zero_upper_kernel (DEVIATION 1640)
        for i in range(n):
            for j in range(i + 1, n):
                a[i * n + j] = Float32(0.0)
        # copy_vector_from_matrix_diagonal_kernel, then the logdet order
        # (logdet_part_kernel, logdet_kernel: cholesky/logdet_fold.mojo)
        var dg = List[Float32](length=max(n, 1), fill=Float32(0.0))
        for j in range(n):
            dg[j] = a[j * n + j]
        logdet = logdet_serial(MutPointer[Float32, MutAnyOrigin](unsafe_from_address=Int(dg.unsafe_ptr())), n)
        _ = dg^
    return CholHostFactor(a^, n, info, logdet, nb, jitter)


#: The right-hand sides a vector solve takes at once: CHOL_TB vectors of
#: CHOL_W columns, so a block's rows of `b` stay in cache while k walks.
comptime CHOL_TB = 8


def _chol_trsm_cols_v(
    l: List[Float32], mut b: List[Float32], n: Int, nrhs: Int, upper: Bool,
) -> Int:
    """`chol_host_trsm_lower` (or `_upper`) for the leading whole CHOL_W
    column groups of `b`, CHOL_W right-hand sides per register: lane `l` of
    a register is column j of the scalar solve, statement for statement
    (the rows in the scalar order, `t = ftz(fma(-ftz(L), ftz(b_k), t))` over
    k in the scalar order, then `identical_div` = `ftz(ftz(t) / ftz(L_ii))`
    lane-wise). Columns never read each other, so solving them side by side
    moves no bit. Returns the first column left to the scalar loop."""
    var whole = (nrhs // CHOL_W) * CHOL_W
    if whole == 0:
        return 0
    var lp = rebind[CholPtr](l.unsafe_ptr())
    var bp = rebind[CholPtr](b.unsafe_ptr())
    var jb = 0
    while jb < whole:
        var nv = min(CHOL_TB, (whole - jb) // CHOL_W)
        for ii in range(n):
            var i = n - 1 - ii if upper else ii
            var lii = ftz(lp.unsafe_load(i * n + i))
            for v in range(nv):
                var col = jb + v * CHOL_W
                var t = ftz_v[CHOL_W](bp.unsafe_load[width=CHOL_W](i * nrhs + col))
                if upper:
                    for k in range(i + 1, n):
                        var lki = ftz(lp.unsafe_load(k * n + i))
                        var bk = ftz_v[CHOL_W](bp.unsafe_load[width=CHOL_W](k * nrhs + col))
                        t = ftz_v[CHOL_W](identical_mul_add_simd[CHOL_W](CholV(-lki), bk, t))
                else:
                    for k in range(i):
                        var lik = ftz(lp.unsafe_load(i * n + k))
                        var bk = ftz_v[CHOL_W](bp.unsafe_load[width=CHOL_W](k * nrhs + col))
                        t = ftz_v[CHOL_W](identical_mul_add_simd[CHOL_W](CholV(-lik), bk, t))
                var q = ftz_v[CHOL_W](ftz_v[CHOL_W](t) / CholV(ftz(lii)))
                bp.unsafe_store(i * nrhs + col, q)
        jb += nv * CHOL_W
    return whole


def chol_host_trsm_lower(
    l: List[Float32], mut b: List[Float32], n: Int, nrhs: Int
):
    """`trsm_lower_kernel` at `ld == n`, in place over `b` (`n x nrhs`
    row-major). One right-hand side at a time; `k` ascending."""
    var j0 = 0
    comptime if not CHOL_HOST_SCALAR:
        j0 = _chol_trsm_cols_v(l, b, n, nrhs, False)
    for j in range(j0, nrhs):
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
    var j0 = 0
    comptime if not CHOL_HOST_SCALAR:
        j0 = _chol_trsm_cols_v(l, b, n, nrhs, True)
    for j in range(j0, nrhs):
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
