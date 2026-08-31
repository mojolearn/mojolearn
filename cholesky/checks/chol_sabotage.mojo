# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Cholesky kernels with their pins BROKEN ON PURPOSE, for the check.

NOT A PORT, NOT REACHED by any driver, by `cholesky/estimator.mojo` or by the
card. Copies of `cholesky/checks/potrf.mojo`'s panel kernel and of
`cholesky/checks/trsm.mojo`'s two solve kernels, each carrying arms that
`cholesky_check.mojo` selects through the `sabotage` argument threaded down
from `potrf_lower` / `cho_solve` and that nothing else can reach. The
un-sabotaged arm of every kernel HERE is never launched: the drivers call the
real kernel when `sabotage == CHOL_SAB_NONE` and one of these otherwise, so
the shipped bits never depend on this file.

Same construction as `hierarchy/checks/sabotage_tile.mojo`, and for the
same reason: a sabotage arm does not belong in a production kernel, and a
sabotage that requires editing source cannot be run by an orchestrator that
is forbidden to edit source.

DEVIATION 1642.

THE ARMS, and what each is a plausible way to get wrong
--------------------------------------------------------
`CHOL_SAB_NB_FROM_LAUNCH`   DRIVER arm (not a kernel here). The block size
                            reaches the numerical parameter: `NB` is
                            derived from `panel_tpb` instead of pinned.
                            This is the defect DEVIATION 1630 exists to
                            forbid, and it is the one a plausible
                            implementation reaches by accident, because
                            "one panel per block, one column per thread"
                            makes `NB = panel_tpb` look like a tidy choice.
`CHOL_SAB_PANEL_DESCENDING` the within-column update walks `k` DESCENDING.
                            A different summation order, same multiset.
`CHOL_SAB_PANEL_ROTATE`     the update starts at `k0 = block_idx.x % width`
                            and wraps, so the order is a function of LAUNCH
                            GEOMETRY. At one block this is inert, which is
                            the point: only the launch gate can see it.
`CHOL_SAB_STD_SQRT`         `std.math.sqrt` on the pivot instead of
                            `identical_sqrt`. Apple's and AMD's are
                            correctly rounded so this may move no bit
                            there; NVIDIA's PTX sqrt is off by one ulp on
                            180,714 of 2^20 patterns (DEVIATION 258).
`CHOL_SAB_NO_FTZ_PIVOT`     `ftz` dropped on the pivot value. Metal flushes
                            in hardware, so this is expected INERT on
                            Apple; on a denormal-honoring column it flips
                            the FIX_DENORMAL_PIVOT fixture from a refusal
                            to a success, which is the divergent-outcome
                            failure the lane exists to forbid.
`CHOL_SAB_PIVOT_GE`         the pivot test becomes `s < 0` instead of
                            `not (s > 0)`, so an exactly-zero pivot passes
                            and the factorization divides by zero.
`CHOL_SAB_VENDOR_MATMUL`    DRIVER arm. The trailing update calls
                            `core/gemm.mojo::gemm_nt` (MAX `linalg.matmul`,
                            a CLOSED library whose k-split is a per-vendor
                            summation order) instead of
                            `identical_gemm_into`.
`CHOL_SAB_TRSM_RECIPROCAL`  the solves multiply by `1 / L[i][i]` instead of
                            dividing by it. This is RAFT's own
                            `getDiagonalInverseMatrix` shape
                            (`matrix/detail/matrix.cuh:283-295`) and a
                            reciprocal-then-multiply is two roundings where
                            a divide is one.
`CHOL_SAB_LOGDET_PAIRWISE`  the log-determinant folds PAIRWISE instead of
                            ascending serial. Same multiset, different
                            order.
`CHOL_SAB_JITTER_RELATIVE`  DRIVER arm. The ridge becomes
                            `jitter * A[i][i]` -- the policy a caller would
                            plausibly choose for itself, and the reason
                            DEVIATION 1637 takes the choice away.

The three DRIVER arms have no kernel in this file; they are branches in
`potrf.mojo`'s driver taken only when `sabotage` names them, exactly as
`hierarchy/impl/cluster/detail/connectivities.mojo` leaves its tie-break,
sort and NaN-guard arms on the production path.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.math import sqrt
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.numerics import (
    ftz,
    identical_div,
    identical_log,
    identical_mul,
    identical_mul_add,
    identical_sqrt,
)


#: The production path. Every driver default.
comptime CHOL_SAB_NONE = 0
comptime CHOL_SAB_NB_FROM_LAUNCH = 1
comptime CHOL_SAB_PANEL_DESCENDING = 2
comptime CHOL_SAB_PANEL_ROTATE = 3
comptime CHOL_SAB_STD_SQRT = 4
comptime CHOL_SAB_NO_FTZ_PIVOT = 5
comptime CHOL_SAB_PIVOT_GE = 6
comptime CHOL_SAB_VENDOR_MATMUL = 7
comptime CHOL_SAB_TRSM_RECIPROCAL = 8
comptime CHOL_SAB_LOGDET_PAIRWISE = 9
comptime CHOL_SAB_JITTER_RELATIVE = 10
comptime CHOL_SAB_COUNT = 11


def chol_sabotage_name(sab: Int) -> String:
    """The arm's name, for the check's banner and for an error message."""
    if sab == CHOL_SAB_NONE:
        return String("NONE")
    if sab == CHOL_SAB_NB_FROM_LAUNCH:
        return String("NB_FROM_LAUNCH")
    if sab == CHOL_SAB_PANEL_DESCENDING:
        return String("PANEL_DESCENDING")
    if sab == CHOL_SAB_PANEL_ROTATE:
        return String("PANEL_ROTATE")
    if sab == CHOL_SAB_STD_SQRT:
        return String("STD_SQRT")
    if sab == CHOL_SAB_NO_FTZ_PIVOT:
        return String("NO_FTZ_PIVOT")
    if sab == CHOL_SAB_PIVOT_GE:
        return String("PIVOT_GE")
    if sab == CHOL_SAB_VENDOR_MATMUL:
        return String("VENDOR_MATMUL")
    if sab == CHOL_SAB_TRSM_RECIPROCAL:
        return String("TRSM_RECIPROCAL")
    if sab == CHOL_SAB_LOGDET_PAIRWISE:
        return String("LOGDET_PAIRWISE")
    if sab == CHOL_SAB_JITTER_RELATIVE:
        return String("JITTER_RELATIVE")
    return String("UNKNOWN")


def chol_sabotage_is_kernel_arm(sab: Int) -> Bool:
    """True when the arm lives in a kernel HERE, false when it is a branch in
    `potrf.mojo`'s driver. The driver uses this to decide which kernel to
    launch, so the production kernel is reached at every driver arm."""
    if sab == CHOL_SAB_PANEL_DESCENDING:
        return True
    if sab == CHOL_SAB_PANEL_ROTATE:
        return True
    if sab == CHOL_SAB_STD_SQRT:
        return True
    if sab == CHOL_SAB_NO_FTZ_PIVOT:
        return True
    if sab == CHOL_SAB_PIVOT_GE:
        return True
    if sab == CHOL_SAB_TRSM_RECIPROCAL:
        # The panel SOLVE carries the reciprocal arm, so the driver has to
        # route the panel through this file for it. The panel FACTOR kernel
        # here has no reciprocal arm, so routing it costs nothing: its
        # un-named arms are the production arithmetic character for
        # character.
        return True
    return False


def sabotage_jitter_diag_kernel(
    a: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    jitter: Float32,
    sabotage_in: Int32,
):
    """`potrf.mojo::jitter_diag_kernel` with the RELATIVE arm.

    The relative policy `A[i,i] += jitter * A[i,i]` is not a wrong idea --
    it is what a GP library usually does, because an absolute ridge is
    meaningless without knowing the kernel's scale. It is refused here
    because it makes the RIDGE a function of the DATA, so two callers with
    the same nominal `jitter` get two different matrices factored, and the
    profile's answer stops being a function of its stated inputs.
    DEVIATION 1637.
    """
    var n = Int(n_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= n:
        return
    var d = ftz(a.unsafe_load(i * n + i))
    if Int(sabotage_in) == CHOL_SAB_JITTER_RELATIVE:
        a.unsafe_store(i * n + i, ftz(identical_mul_add(jitter, d, d)))
        return
    a.unsafe_store(i * n + i, ftz(d + jitter))


def sabotage_panel_factor_kernel(
    a: MutPointer[Float32, MutAnyOrigin],
    info: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
    j0_in: Int32,
    nb_in: Int32,
    sabotage_in: Int32,
):
    """`potrf.mojo::panel_factor_kernel` with five arms. ONE block.

    Everything not named by an arm is character for character the
    production kernel, so a failure names the arm and nothing else.
    """
    var n = Int(n_in)
    var j0 = Int(j0_in)
    var nb = Int(nb_in)
    var sab = Int(sabotage_in)
    var tid = Int(thread_idx.x)
    var width = Int(block_dim.x)

    var flag = stack_allocation[
        1, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    if tid == 0:
        flag[0] = Int32(0)
    barrier()

    for c in range(nb):
        var jc = j0 + c
        if tid == 0:
            var s = ftz(a.unsafe_load(jc * n + jc))
            if sab == CHOL_SAB_NO_FTZ_PIVOT:
                # ARM: the seam flush is dropped. Bitwise inert wherever the
                # hardware already flushes.
                s = a.unsafe_load(jc * n + jc)
            for kk in range(jc - j0):
                var k = j0 + kk
                if sab == CHOL_SAB_PANEL_DESCENDING:
                    k = jc - 1 - kk
                var v = ftz(a.unsafe_load(jc * n + k))
                s = ftz(identical_mul_add(-v, v, s))
            var bad = not (s > Float32(0.0))
            if sab == CHOL_SAB_PIVOT_GE:
                # ARM: an exactly-zero pivot passes; a NaN pivot passes too.
                bad = s < Float32(0.0)
            if bad:
                info.unsafe_store(0, Int32(jc + 1))
                flag[0] = Int32(1)
            else:
                if sab == CHOL_SAB_STD_SQRT:
                    a.unsafe_store(jc * n + jc, ftz(sqrt(s)))
                else:
                    a.unsafe_store(jc * n + jc, ftz(identical_sqrt(s)))
        barrier()
        if flag[0] != Int32(0):
            return
        var ljj = ftz(a.unsafe_load(jc * n + jc))
        var i = c + 1 + tid
        while i < nb:
            var r = j0 + i
            var t = ftz(a.unsafe_load(r * n + jc))
            var span = jc - j0
            var start = 0
            if sab == CHOL_SAB_PANEL_ROTATE and span > 0:
                # ARM: the order is a function of the physical block index.
                start = Int(block_idx.x) % span
            for kk in range(span):
                var off = start + kk
                if off >= span:
                    off -= span
                if sab == CHOL_SAB_PANEL_DESCENDING:
                    off = span - 1 - kk
                var k = j0 + off
                var lrk = ftz(a.unsafe_load(r * n + k))
                var lck = ftz(a.unsafe_load(jc * n + k))
                t = ftz(identical_mul_add(-lrk, lck, t))
            a.unsafe_store(r * n + jc, ftz(identical_div(t, ljj)))
            i += width
        barrier()


def sabotage_trsm_panel_kernel(
    a: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    j0_in: Int32,
    nb_in: Int32,
    n_trail_in: Int32,
    sabotage_in: Int32,
):
    """`trsm.mojo::trsm_panel_kernel` with the ordering and reciprocal arms.
    One thread per trailing row, as the production kernel is."""
    var n = Int(n_in)
    var j0 = Int(j0_in)
    var nb = Int(nb_in)
    var n_trail = Int(n_trail_in)
    var sab = Int(sabotage_in)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= n_trail:
        return
    var r = j0 + nb + idx

    for c in range(nb):
        var jc = j0 + c
        var t = ftz(a.unsafe_load(r * n + jc))
        var span = c
        var start = 0
        if sab == CHOL_SAB_PANEL_ROTATE and span > 0:
            start = Int(block_idx.x) % span
        for kk in range(span):
            var off = start + kk
            if off >= span:
                off -= span
            if sab == CHOL_SAB_PANEL_DESCENDING:
                off = span - 1 - kk
            var k = j0 + off
            var lrk = ftz(a.unsafe_load(r * n + k))
            var lck = ftz(a.unsafe_load(jc * n + k))
            t = ftz(identical_mul_add(-lrk, lck, t))
        var ljj = ftz(a.unsafe_load(jc * n + jc))
        if sab == CHOL_SAB_TRSM_RECIPROCAL:
            var inv = ftz(identical_div(Float32(1.0), ljj))
            a.unsafe_store(r * n + jc, ftz(identical_mul(t, inv)))
        else:
            a.unsafe_store(r * n + jc, ftz(identical_div(t, ljj)))


def sabotage_trsm_lower_kernel(
    l: MutPointer[Float32, MutAnyOrigin],
    b: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    nrhs_in: Int32,
    ld_in: Int32,
    sabotage_in: Int32,
):
    """`trsm.mojo::trsm_lower_kernel` with the reciprocal and ordering arms.
    One thread per right-hand-side COLUMN."""
    var n = Int(n_in)
    var nrhs = Int(nrhs_in)
    var ld = Int(ld_in)
    var sab = Int(sabotage_in)
    var j = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if j >= nrhs:
        return

    for i in range(n):
        var t = ftz(b.unsafe_load(i * nrhs + j))
        for kk in range(i):
            var k = kk
            if sab == CHOL_SAB_PANEL_DESCENDING:
                k = i - 1 - kk
            var lik = ftz(l.unsafe_load(i * ld + k))
            var bk = ftz(b.unsafe_load(k * nrhs + j))
            t = ftz(identical_mul_add(-lik, bk, t))
        var lii = ftz(l.unsafe_load(i * ld + i))
        if sab == CHOL_SAB_TRSM_RECIPROCAL:
            var inv = ftz(identical_div(Float32(1.0), lii))
            b.unsafe_store(i * nrhs + j, ftz(identical_mul(t, inv)))
        else:
            b.unsafe_store(i * nrhs + j, ftz(identical_div(t, lii)))


def sabotage_trsm_upper_kernel(
    l: MutPointer[Float32, MutAnyOrigin],
    b: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    nrhs_in: Int32,
    ld_in: Int32,
    sabotage_in: Int32,
):
    """`trsm.mojo::trsm_upper_kernel` with the same two arms."""
    var n = Int(n_in)
    var nrhs = Int(nrhs_in)
    var ld = Int(ld_in)
    var sab = Int(sabotage_in)
    var j = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if j >= nrhs:
        return

    for ii in range(n):
        var i = n - 1 - ii
        var t = ftz(b.unsafe_load(i * nrhs + j))
        var span = n - 1 - i
        for kk in range(span):
            var k = i + 1 + kk
            if sab == CHOL_SAB_PANEL_DESCENDING:
                k = n - 1 - kk
            var lki = ftz(l.unsafe_load(k * ld + i))
            var bk = ftz(b.unsafe_load(k * nrhs + j))
            t = ftz(identical_mul_add(-lki, bk, t))
        var lii = ftz(l.unsafe_load(i * ld + i))
        if sab == CHOL_SAB_TRSM_RECIPROCAL:
            var inv = ftz(identical_div(Float32(1.0), lii))
            b.unsafe_store(i * nrhs + j, ftz(identical_mul(t, inv)))
        else:
            b.unsafe_store(i * nrhs + j, ftz(identical_div(t, lii)))


def sabotage_logdet_kernel(
    diag: MutPointer[Float32, MutAnyOrigin],
    out_scalar: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    sabotage_in: Int32,
):
    """`potrf.mojo::logdet_kernel` with the PAIRWISE arm. ONE thread.

    Both arms read the already-extracted diagonal, as the production kernel
    does. The pairwise arm writes its logs back over `diag` and folds them in
    place, so it needs no scratch the production kernel does not have; the
    caller has already recorded `chol.diag`, so overwriting it after the
    record moves no card stage.
    """
    if Int(block_idx.x) != 0 or Int(thread_idx.x) != 0:
        return
    var n = Int(n_in)
    var sab = Int(sabotage_in)

    if sab == CHOL_SAB_LOGDET_PAIRWISE:
        for j in range(n):
            diag.unsafe_store(j, ftz(identical_log(ftz(diag.unsafe_load(j)))))
        var width = n
        while width > 1:
            var half = (width + 1) // 2
            for t in range(width - half):
                var x = ftz(diag.unsafe_load(t))
                var y = ftz(diag.unsafe_load(t + half))
                diag.unsafe_store(t, ftz(x + y))
            width = half
        var acc_p = Float32(0.0)
        if n > 0:
            acc_p = ftz(diag.unsafe_load(0))
        out_scalar.unsafe_store(0, ftz(identical_mul(Float32(2.0), acc_p)))
        return

    var acc = Float32(0.0)
    for j in range(n):
        acc = ftz(acc + ftz(identical_log(ftz(diag.unsafe_load(j)))))
    out_scalar.unsafe_store(0, ftz(identical_mul(Float32(2.0), acc)))
