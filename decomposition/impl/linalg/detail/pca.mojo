# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""PCA by covariance eigendecomposition. The `input`-unchanged CONTRACT is the one to not drop: `input` is an in-out parameter that must end the call unchanged, and a fit that leaves the caller's matrix centered is wrong in a way nothing in the fit itself will reveal."""

from std.gpu import block_dim, block_idx, thread_idx
from std.math import sqrt
from std.os import getenv
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from core.pinned_reduce import (
    pinned_block_max as block_max,
    pinned_block_min as block_min,
)
from max.gpu.sync import barrier
from std.memory import stack_allocation

from core.gemm import gemm_nt, gemm_tn
from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_div,
    identical_mul,
)
from core.device_zero import enqueue_fill
from x_decomp.jacobi_par import (
    PJ_TPB,
    eigh_par_cs_kernel,
    eigh_par_off_kernel,
    eigh_par_update_kernel,
    pj_identity_kernel,
)
from core.gram_splitk import gram_centered_splitk_into, gram_splitk_applies
from core.column_stats import (
    STATS_TPB,
    column_mean_kernel,
    scale_in_place_kernel,
    shift_columns_kernel,
)
from decomposition.checks.jacobi_eigh import jacobi_eigh
from decomposition.checks.jacobi_eigh_device import (
    JACOBI_SWEEPS,
    JACOBI_TOL,
    JACOBI_ROT_TPB,
    jacobi_eigh_kernel,
)


@fieldwise_init
struct PCAResult(Movable):
    """What `pca_fit` writes back on the host side."""

    var components: List[Float64]
    var explained_var: List[Float64]
    var explained_var_ratio: List[Float64]
    var singular_vals: List[Float64]
    var noise_var: Float64


def compute_covariance(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    mut x_alias: DeviceBuffer[DType.float32],
    mut x_alias2: DeviceBuffer[DType.float32],
    mut mu: DeviceBuffer[DType.float32],
    mut cov: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_cols: Int,
    restore_input: Bool = True,
) raises:
    """Steps 1, 2, 3 and 6. The branch below must take the fused arm exactly when `gemm_tn` would take split-K for this shape, so it asks the SAME `gram_splitk_applies(m, n, k)` that `gemm_tn` asks -- one predicate, both readers, no target test of our own."""
    ctx.enqueue_function[column_mean_kernel](
        mu.unsafe_ptr(),
        x.unsafe_ptr(),
        Int32(n_rows),
        Int32(n_cols),
        grid_dim=(n_cols, 1, 1),
        block_dim=(STATS_TPB, 1, 1),
    )
    var cells = n_rows * n_cols
    var fused = gram_splitk_applies(n_cols, n_cols, n_rows)
    if fused:
        gram_centered_splitk_into(
            ctx, cov, x, mu, x_alias, n_cols, n_rows
        )
    else:
        ctx.enqueue_function[shift_columns_kernel](
            x.unsafe_ptr(),
            mu.unsafe_ptr(),
            Int32(n_rows),
            Int32(n_cols),
            Float32(-1.0),
            grid_dim=((cells + 255) // 256, 1, 1),
            block_dim=(256, 1, 1),
        )
        gemm_tn(ctx, cov, x, x_alias, x_alias2, n_cols, n_cols, n_rows)
    ctx.enqueue_function[scale_in_place_kernel](
        cov.unsafe_ptr(),
        Int32(n_cols * n_cols),
        Float32(1.0) / Float32(n_rows - 1),
        grid_dim=((n_cols * n_cols + 255) // 256, 1, 1),
        block_dim=(256, 1, 1),
    )
    if restore_input and not fused:
        ctx.enqueue_function[shift_columns_kernel](
            x.unsafe_ptr(),
            mu.unsafe_ptr(),
            Int32(n_rows),
            Int32(n_cols),
            Float32(1.0),
            grid_dim=((cells + 255) // 256, 1, 1),
            block_dim=(256, 1, 1),
        )
    ctx.synchronize()


def pca_validate(n_rows: Int, n_cols: Int, n_components: Int) raises:
    """The four shape refusals `pca_fit` makes first (cuML's `pcaFit` asserts, in their order)."""
    if n_cols <= 1:
        raise Error("Parameter n_cols: number of columns cannot be less than two")
    if n_rows <= 1:
        raise Error("Parameter n_rows: number of rows cannot be less than two")
    if n_components <= 0:
        raise Error(
            "Parameter n_components: number of components cannot be less than one"
        )
    if n_components > n_cols:
        raise Error("n_components cannot exceed n_cols")


def pca_fit(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    mut x_alias: DeviceBuffer[DType.float32],
    mut x_alias2: DeviceBuffer[DType.float32],
    mut mu: DeviceBuffer[DType.float32],
    mut cov: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_cols: Int,
    n_components: Int,
    restore_input: Bool = True,
) raises -> PCAResult:
    """`pca_fit`, all six steps."""
    pca_validate(n_rows, n_cols, n_components)

    compute_covariance(
        ctx, x, x_alias, x_alias2, mu, cov, n_rows, n_cols, restore_input
    )

    return eig_and_truncate(
        ctx, cov, n_cols, n_components, n_rows - 1
    )


comptime SIGNFLIP_TPB = 32


def sign_flip_kernel(
    v: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """Reference: `signFlipKernel`, `raft/matrix/detail/math.cuh:367`. `decomposition/checks/pca_check.mojo` holds it to that: the device answer must equal a fold-free host scan BITWISE, and the tie, the zero cases and the NaN case are each planted rather than hoped for."""
    var n = Int(n_in)
    var col = Int(block_idx.x)
    var tid = Int(thread_idx.x)

    var sh = stack_allocation[
        3,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()

    var local = Float32(0.0)
    var f = tid
    while f < n:
        var m = abs(v.unsafe_load(f * n + col))
        if m > local:
            local = m
        f += SIGNFLIP_TPB
    var reduced_max = block_max[block_size=SIGNFLIP_TPB](local)
    if tid == 0:
        sh[0] = reduced_max
    barrier()
    var biggest = sh[0]

    var cand = Float32(n)
    f = tid
    while f < n:
        var fv = Float32(f)
        if abs(v.unsafe_load(f * n + col)) == biggest:
            if fv < cand:
                cand = fv
        f += SIGNFLIP_TPB
    var reduced_first = block_min[block_size=SIGNFLIP_TPB](cand)
    if tid == 0:
        sh[1] = reduced_first
        sh[2] = Float32(1.0)
    barrier()
    var first = sh[1]

    f = tid
    while f < n:
        if Float32(f) == first:
            if v.unsafe_load(f * n + col) < Float32(0.0):
                sh[2] = Float32(-1.0)
            else:
                sh[2] = Float32(1.0)
        f += SIGNFLIP_TPB
    barrier()

    if sh[2] < Float32(0.0):
        f = tid
        while f < n:
            v.unsafe_store(f * n + col, -v.unsafe_load(f * n + col))
            f += SIGNFLIP_TPB


def order_truncate_spectrum(
    diag: List[Float64],
    vecs: List[Float64],
    n_cols: Int,
    n_components: Int,
    singular_scale: Int,
    spectrum_count: Int = 0,
) raises -> PCAResult:
    """`colReverse` + `truncCompExpVars`: the HOST tail of a fit, shared."""

    var count = n_cols if spectrum_count == 0 else spectrum_count
    var order = List[Int]()
    for i in range(count):
        order.append(i)
    for i in range(count):
        for j in range(i + 1, count):
            if diag[order[j]] > diag[order[i]]:
                var t = order[i]
                order[i] = order[j]
                order[j] = t

    var total = 0.0
    for i in range(count):
        total += diag[i]

    var components = List[Float64]()
    var explained_var = List[Float64]()
    var explained_var_ratio = List[Float64]()
    var singular_vals = List[Float64]()

    for c in range(n_components):
        var src = order[c]
        var lam = diag[src]

        for f in range(n_cols):
            components.append(vecs[f * n_cols + src])
        explained_var.append(lam)
        explained_var_ratio.append(lam / total if total != 0.0 else 0.0)
        singular_vals.append(sqrt(lam * Float64(singular_scale)))

    var noise = 0.0
    if n_components < count and n_components <= singular_scale:
        for c in range(n_components, count):
            noise += diag[order[c]]
        noise /= Float64(count - n_components)

    return PCAResult(
        components^, explained_var^, explained_var_ratio^, singular_vals^, noise
    )

# ---------------------------------------------------------------------------
# lane/apple-fast-pca-eig (2026-10-02): the eigen step of PCA / TruncatedSVD
# on the grid, FAST only, two env switches read here on the host at dispatch.
#
# CAUSE. `eig_and_truncate` below launches `jacobi_eigh_kernel` with
# grid_dim=(1, 1, 1): one block of 256 threads does every one of the
# n (n - 1) / 2 rotations of a sweep in cyclic order behind a barrier
# (24,090 rotations a sweep at Istella's 220 columns, up to JACOBI_SWEEPS
# sweeps) while the rest of the GPU idles; then it copies the whole n x n
# covariance and the whole n x n basis to the host and orders, truncates and
# gathers the n_components columns there (`order_truncate_spectrum`). Board:
# pca Istella FAST 988 ms vs IDENTICAL 814 vs best opponent ~206.
#
#   MOJOLEARN_PCA_FAST_EIG=1   the round-robin Jacobi of x_decomp/jacobi_par.mojo
#       (the x_decomp kit's own default eigh order since lane/neural-pass104):
#       m - 1 rounds a sweep, each round's m / 2 disjoint rotations across the
#       grid in two launches (`eigh_par_cs_kernel`, `eigh_par_update_kernel`),
#       the sweep test (the cyclic kernel's: off <= tol^2 ||A||_F^2) folded on
#       the device and read back as four floats once a sweep. A solve that does
#       not converge in PCA_FAST_RR_SWEEPS sweeps, or does not keep ||A||_F,
#       leaves `cov` untouched and the cyclic kernel below runs as before, so
#       the convergence refusal keeps its name and its text.
#   MOJOLEARN_PCA_FAST_TOPK=1  the descending order of the n eigenvalues
#       (`pf_rank_kernel`, one thread per value, rank by count) and the gather
#       of the n_components columns of V (`pf_gather_kernel`) on the device:
#       n + n_components x n floats come back instead of 2 n x n.
#
# NOT an IDENTICAL path: both switches sit under
# `comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL` and default off, so
# IDENTICAL compiles and runs the code below them unchanged. Written without
# a Mojo toolchain; the first M3 build is the compile check.
# ---------------------------------------------------------------------------

comptime PF_TPB = 256
"""Launch width of the FAST tail kernels here (PJ_TPB's value)."""

comptime PCA_FAST_RR_SWEEPS = 30
"""Sweep budget of the round-robin solve (x_decomp/device.mojo PJ_EIGH_SWEEPS)."""


def pca_fast_eig_on() -> Bool:
    """MOJOLEARN_PCA_FAST_EIG=1: the round-robin eigen step (FAST only)."""
    return String(getenv("MOJOLEARN_PCA_FAST_EIG")) == "1"


def pca_fast_topk_on() -> Bool:
    """MOJOLEARN_PCA_FAST_TOPK=1: order and gather on the device (FAST only)."""
    return String(getenv("MOJOLEARN_PCA_FAST_TOPK")) == "1"


def _pf_blocks(count: Int) -> Int:
    return (count + PF_TPB - 1) // PF_TPB if count > 0 else 1


def pf_sweep_test_kernel(
    rows: MutPointer[Float32, MutAnyOrigin],
    test_out: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    tol_in: Float32,
):
    """The round-robin sweep test on the device: `rows` is
    `eigh_par_off_kernel`'s output (rows[k] = row k's off-diagonal squares,
    rows[n + k] = a_kk^2). test_out[0] = off, [1] = ||A||_F^2 (off + diag),
    [2] = 1.0 when off <= tol^2 ||A||_F^2, [3] = sqrt(off / ||A||_F^2).
    One block of PF_TPB threads, 256 partials, a halving tree."""
    var n = Int(n_in)
    var tid = Int(thread_idx.x)
    var slab = stack_allocation[
        2 * PF_TPB,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var off = Float32(0.0)
    var dg = Float32(0.0)
    var k = tid
    while k < n:
        off = rows.unsafe_load(k) + off
        dg = rows.unsafe_load(n + k) + dg
        k += PF_TPB
    slab[tid] = off
    slab[PF_TPB + tid] = dg
    barrier()
    if tid == 0:
        var w = PF_TPB // 2
        while w > 0:
            for t in range(w):
                slab[t] = slab[t] + slab[t + w]
                slab[PF_TPB + t] = slab[PF_TPB + t] + slab[PF_TPB + t + w]
            w = w // 2
        var o = slab[0]
        var f = slab[0] + slab[PF_TPB]
        test_out.unsafe_store(0, o)
        test_out.unsafe_store(1, f)
        var ok = Float32(0.0)
        if o <= tol_in * tol_in * f:
            ok = Float32(1.0)
        test_out.unsafe_store(2, ok)
        var rel = Float32(0.0)
        if f > Float32(0.0):
            rel = sqrt(o / f)
        test_out.unsafe_store(3, rel)


def pf_rank_kernel(
    a: MutPointer[Float32, MutAnyOrigin],
    diag_out: MutPointer[Float32, MutAnyOrigin],
    order_out: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
):
    """order_out[r] = the index of the eigenvalue of descending rank r
    (a_ii from the diagonalized `a`; ties to the lower index), diag_out[i] =
    a_ii. One thread per value, rank by count. A non-finite value compares
    false everywhere, so its slot stays at the -1 the host filled and the
    host refuses the spectrum. Launch with ceil(n / PF_TPB) blocks."""
    var n = Int(n_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < n:
        var di = a.unsafe_load(i * n + i)
        var rank = 0
        for j in range(n):
            var dj = a.unsafe_load(j * n + j)
            if dj > di or (dj == di and j < i):
                rank += 1
        diag_out.unsafe_store(i, di)
        if rank < n:
            order_out.unsafe_store(rank, Int32(i))


def pf_gather_kernel(
    v: MutPointer[Float32, MutAnyOrigin],
    order: MutPointer[Int32, MutAnyOrigin],
    comp_out: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    k_in: Int32,
):
    """comp_out[c * n + f] = v[f * n + order[c]] for c < k: the n_components
    columns of V, each as a row of the components matrix (what
    `order_truncate_spectrum` builds on the host). One thread per cell."""
    var n = Int(n_in)
    var cells = Int(k_in) * n
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < cells:
        var c = t // n
        var f = t - c * n
        var src = Int(order.unsafe_load(c))
        var val = Float32(0.0)
        if src >= 0 and src < n:
            val = v.unsafe_load(f * n + src)
        comp_out.unsafe_store(t, val)


def _pca_fast_rr_eigh(
    ctx: DeviceContext,
    mut cov: DeviceBuffer[DType.float32],
    mut vec_buf: DeviceBuffer[DType.float32],
    n: Int,
) raises -> SIMD[DType.float32, 4]:
    """The round-robin Jacobi of x_decomp/jacobi_par.mojo on a copy of
    `cov` (x_decomp/device.mojo `_eigh_par`'s rounds, the sweep test on the
    device). Returns (converged, rel, sweeps run, 0). Converged: `cov` is
    the diagonalized matrix (its diagonal the eigenvalues) and `vec_buf` the
    basis V, exactly what `jacobi_eigh_kernel` leaves. Not converged: `cov`
    is untouched and the caller runs the cyclic kernel on it."""
    var m = n + (n % 2)
    var h = m // 2
    var cells = n * n
    var a_work = ctx.enqueue_create_buffer[DType.float32](cells)
    ctx.enqueue_copy(dst_buf=a_work, src_buf=cov)
    var dcs = ctx.enqueue_create_buffer[DType.float32](2 * h)
    var doff = ctx.enqueue_create_buffer[DType.float32](3 * n)
    var dtest = ctx.enqueue_create_buffer[DType.float32](4)
    var htest = ctx.enqueue_create_host_buffer[DType.float32](4)
    ctx.enqueue_function[pj_identity_kernel](
        vec_buf.unsafe_ptr(),
        Int32(n),
        grid_dim=_pf_blocks(cells),
        block_dim=PJ_TPB,
    )
    var converged = False
    var executed = 0
    var fro_in = Float32(-1.0)
    var fro_now = Float32(0.0)
    var rel = Float32(0.0)
    for sweep in range(PCA_FAST_RR_SWEEPS + 1):
        # a sum of squares is never negative: -1 left in the readback is a
        # dispatch that did not run (x_decomp/device.mojo `_eigh_par`)
        enqueue_fill(ctx, dtest, Float32(-1.0))
        ctx.enqueue_function[eigh_par_off_kernel](
            a_work.unsafe_ptr(),
            doff.unsafe_ptr(),
            Int32(n),
            grid_dim=_pf_blocks(n),
            block_dim=PJ_TPB,
        )
        ctx.enqueue_function[pf_sweep_test_kernel](
            doff.unsafe_ptr(),
            dtest.unsafe_ptr(),
            Int32(n),
            Float32(JACOBI_TOL),
            grid_dim=(1, 1, 1),
            block_dim=(PF_TPB, 1, 1),
        )
        ctx.enqueue_copy(dst_ptr=htest.unsafe_ptr(), src_buf=dtest)
        ctx.synchronize()
        var off = htest.unsafe_ptr().unsafe_load(0)
        fro_now = htest.unsafe_ptr().unsafe_load(1)
        rel = htest.unsafe_ptr().unsafe_load(3)
        if off < Float32(0.0) or fro_now < Float32(0.0):
            break
        if fro_in < Float32(0.0):
            fro_in = fro_now
        if htest.unsafe_ptr().unsafe_load(2) != Float32(0.0):
            converged = True
            break
        if sweep == PCA_FAST_RR_SWEEPS:
            break
        executed += 1
        for rd in range(m - 1):
            ctx.enqueue_function[eigh_par_cs_kernel](
                a_work.unsafe_ptr(),
                dcs.unsafe_ptr(),
                Int32(n),
                Int32(m),
                Int32(rd),
                grid_dim=_pf_blocks(h),
                block_dim=PJ_TPB,
            )
            ctx.enqueue_function[eigh_par_update_kernel](
                a_work.unsafe_ptr(),
                vec_buf.unsafe_ptr(),
                dcs.unsafe_ptr(),
                Int32(n),
                Int32(m),
                Int32(rd),
                grid_dim=_pf_blocks(h * h + n * h),
                block_dim=PJ_TPB,
            )
    # J^T A J keeps ||A||_F: a solve that moved it is not an answer
    if converged and not (abs(fro_now - fro_in) <= Float32(1.0e-3) * fro_in):
        converged = False
    if converged:
        ctx.enqueue_copy(dst_buf=cov, src_buf=a_work)
    ctx.synchronize()
    _ = a_work^
    _ = dcs^
    _ = doff^
    _ = dtest^
    _ = htest^
    return SIMD[DType.float32, 4](
        Float32(1.0) if converged else Float32(0.0),
        rel,
        Float32(executed),
        Float32(0.0),
    )


def _pca_fast_topk_tail(
    ctx: DeviceContext,
    mut cov: DeviceBuffer[DType.float32],
    mut vec_buf: DeviceBuffer[DType.float32],
    mut info_buf: DeviceBuffer[DType.float32],
    n_cols: Int,
    n_components: Int,
    singular_scale: Int,
) raises -> PCAResult:
    """`order_truncate_spectrum`'s order and gather on the device; the
    n-float diagonal, the n-int order and the n_components x n components
    come back, and the host does the same float64 ratios, square roots and
    noise mean as before, in the same order."""
    var order_buf = ctx.enqueue_create_buffer[DType.int32](n_cols)
    enqueue_fill(ctx, order_buf, Int32(-1))
    var diag_buf = ctx.enqueue_create_buffer[DType.float32](n_cols)
    var comp_buf = ctx.enqueue_create_buffer[DType.float32](
        n_components * n_cols
    )
    ctx.enqueue_function[pf_rank_kernel](
        cov.unsafe_ptr(),
        diag_buf.unsafe_ptr(),
        order_buf.unsafe_ptr(),
        Int32(n_cols),
        grid_dim=_pf_blocks(n_cols),
        block_dim=PF_TPB,
    )
    ctx.enqueue_function[pf_gather_kernel](
        vec_buf.unsafe_ptr(),
        order_buf.unsafe_ptr(),
        comp_buf.unsafe_ptr(),
        Int32(n_cols),
        Int32(n_components),
        grid_dim=_pf_blocks(n_components * n_cols),
        block_dim=PF_TPB,
    )
    var h_diag = ctx.enqueue_create_host_buffer[DType.float32](n_cols)
    var h_order = ctx.enqueue_create_host_buffer[DType.int32](n_cols)
    var h_comp = ctx.enqueue_create_host_buffer[DType.float32](
        n_components * n_cols
    )
    var h_info = ctx.enqueue_create_host_buffer[DType.float32](3)
    ctx.enqueue_copy(dst_ptr=h_diag.unsafe_ptr(), src_buf=diag_buf)
    ctx.enqueue_copy(dst_ptr=h_order.unsafe_ptr(), src_buf=order_buf)
    ctx.enqueue_copy(dst_ptr=h_comp.unsafe_ptr(), src_buf=comp_buf)
    ctx.enqueue_copy(dst_ptr=h_info.unsafe_ptr(), src_buf=info_buf)
    ctx.synchronize()

    if h_info.unsafe_ptr().unsafe_load(0) == Float32(0.0):
        raise Error(
            "the device Jacobi did not converge in "
            + String(JACOBI_SWEEPS)
            + " sweeps at n_cols = "
            + String(n_cols)
            + ": ||offdiag(A)||_F / ||A||_F is still "
            + String(h_info.unsafe_ptr().unsafe_load(1))
            + " against a tolerance of "
            + String(JACOBI_TOL)
            + ". cuSOLVER's syevj has the same failure mode and the same"
            " remedy, which is more sweeps. A non-symmetric covariance"
            " produces this too; see check_covariance_is_symmetric."
        )

    var diag = List[Float64]()
    for i in range(n_cols):
        diag.append(Float64(h_diag.unsafe_ptr().unsafe_load(i)))
    var order = List[Int]()
    for i in range(n_cols):
        var o = Int(h_order.unsafe_ptr().unsafe_load(i))
        if o < 0 or o >= n_cols:
            raise Error(
                "the device ordering of the spectrum left rank "
                + String(i)
                + " of "
                + String(n_cols)
                + " unassigned: a non-finite eigenvalue (a non-finite"
                " covariance) has no place in a total order"
            )
        order.append(o)

    var total = 0.0
    for i in range(n_cols):
        total += diag[i]

    var components = List[Float64]()
    var explained_var = List[Float64]()
    var explained_var_ratio = List[Float64]()
    var singular_vals = List[Float64]()

    for c in range(n_components):
        var src = order[c]
        var lam = diag[src]
        for f in range(n_cols):
            components.append(
                Float64(h_comp.unsafe_ptr().unsafe_load(c * n_cols + f))
            )
        explained_var.append(lam)
        explained_var_ratio.append(lam / total if total != 0.0 else 0.0)
        singular_vals.append(sqrt(lam * Float64(singular_scale)))

    var noise = 0.0
    if n_components < n_cols and n_components <= singular_scale:
        for c in range(n_components, n_cols):
            noise += diag[order[c]]
        noise /= Float64(n_cols - n_components)

    _ = order_buf^
    _ = diag_buf^
    _ = comp_buf^
    _ = h_diag^
    _ = h_order^
    _ = h_comp^
    _ = h_info^
    return PCAResult(
        components^, explained_var^, explained_var_ratio^, singular_vals^, noise
    )


def eig_and_truncate(
    ctx: DeviceContext,
    mut cov: DeviceBuffer[DType.float32],
    n_cols: Int,
    n_components: Int,
    singular_scale: Int,
) raises -> PCAResult:
    """`calEig` + `truncCompExpVars`, shared by PCA and truncated SVD."""
    var vec_buf = ctx.enqueue_create_buffer[DType.float32](n_cols * n_cols)
    var info_buf = ctx.enqueue_create_buffer[DType.float32](3)
    ctx.synchronize()
    # lane/apple-fast-pca-eig: MOJOLEARN_PCA_FAST_EIG=1 (FAST only) takes
    # the round-robin solve across the grid; `took_fast` False (not asked,
    # or not converged, `cov` untouched) runs the one-block cyclic kernel
    # below exactly as before. See the block above `pca_fast_eig_on`.
    var took_fast = False
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        if pca_fast_eig_on():
            var got = _pca_fast_rr_eigh(ctx, cov, vec_buf, n_cols)
            if got[0] != Float32(0.0):
                took_fast = True
                var h_set = ctx.enqueue_create_host_buffer[DType.float32](3)
                h_set.unsafe_ptr().unsafe_store(0, Float32(1.0))
                h_set.unsafe_ptr().unsafe_store(1, got[1])
                h_set.unsafe_ptr().unsafe_store(2, got[2])
                ctx.enqueue_copy(dst_buf=info_buf, src_ptr=h_set.unsafe_ptr())
                ctx.synchronize()
                _ = h_set^
    if not took_fast:
        ctx.enqueue_function[jacobi_eigh_kernel[JACOBI_ROT_TPB]](
            cov.unsafe_ptr(),
            vec_buf.unsafe_ptr(),
            info_buf.unsafe_ptr(),
            Int32(n_cols),
            Int32(JACOBI_SWEEPS),
            Float32(JACOBI_TOL),
            grid_dim=(1, 1, 1),
            block_dim=(JACOBI_ROT_TPB, 1, 1),
        )

    ctx.enqueue_function[sign_flip_kernel](
        vec_buf.unsafe_ptr(),
        Int32(n_cols),
        grid_dim=(n_cols, 1, 1),
        block_dim=(SIGNFLIP_TPB, 1, 1),
    )
    ctx.synchronize()

    # lane/apple-fast-pca-eig: MOJOLEARN_PCA_FAST_TOPK=1 (FAST only) orders
    # and gathers on the device and reads back n + n_components x n floats
    # instead of the two n x n matrices below.
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        if pca_fast_topk_on():
            return _pca_fast_topk_tail(
                ctx, cov, vec_buf, info_buf, n_cols, n_components, singular_scale
            )

    var h_cov = ctx.enqueue_create_host_buffer[DType.float32](n_cols * n_cols)
    var h_vec = ctx.enqueue_create_host_buffer[DType.float32](n_cols * n_cols)
    var h_info = ctx.enqueue_create_host_buffer[DType.float32](3)
    ctx.enqueue_copy(dst_ptr=h_cov.unsafe_ptr(), src_buf=cov)
    ctx.enqueue_copy(dst_ptr=h_vec.unsafe_ptr(), src_buf=vec_buf)
    ctx.enqueue_copy(dst_ptr=h_info.unsafe_ptr(), src_buf=info_buf)
    ctx.synchronize()

    if h_info.unsafe_ptr().unsafe_load(0) == Float32(0.0):
        raise Error(
            "the device Jacobi did not converge in "
            + String(JACOBI_SWEEPS)
            + " sweeps at n_cols = "
            + String(n_cols)
            + ": ||offdiag(A)||_F / ||A||_F is still "
            + String(h_info.unsafe_ptr().unsafe_load(1))
            + " against a tolerance of "
            + String(JACOBI_TOL)
            + ". cuSOLVER's syevj has the same failure mode and the same"
            " remedy, which is more sweeps. A non-symmetric covariance"
            " produces this too; see check_covariance_is_symmetric."
        )

    var diag = List[Float64]()
    for i in range(n_cols):
        diag.append(Float64(h_cov.unsafe_ptr().unsafe_load(i * n_cols + i)))
    var vecs = List[Float64]()
    for i in range(n_cols * n_cols):
        vecs.append(Float64(h_vec.unsafe_ptr().unsafe_load(i)))

    return order_truncate_spectrum(
        diag, vecs, n_cols, n_components, singular_scale
    )


def pca_transform(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    mut mu: DeviceBuffer[DType.float32],
    mut components: DeviceBuffer[DType.float32],
    mut out: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_cols: Int,
    n_components: Int,
) raises:
    """`pca_transform`: center, then project onto the components."""
    var cells = n_rows * n_cols
    ctx.enqueue_function[shift_columns_kernel](
        x.unsafe_ptr(),
        mu.unsafe_ptr(),
        Int32(n_rows),
        Int32(n_cols),
        Float32(-1.0),
        grid_dim=((cells + 255) // 256, 1, 1),
        block_dim=(256, 1, 1),
    )
    gemm_nt(
        ctx,
        out,
        x,
        components,
        n_rows,
        n_components,
        n_cols,
    )
    ctx.enqueue_function[shift_columns_kernel](
        x.unsafe_ptr(),
        mu.unsafe_ptr(),
        Int32(n_rows),
        Int32(n_cols),
        Float32(1.0),
        grid_dim=((cells + 255) // 256, 1, 1),
        block_dim=(256, 1, 1),
    )
    ctx.synchronize()




comptime WHITEN_SKIP_ZERO = 1.0e-10


def whiten_scalar(n_fit_rows: Int, inverse: Bool) -> Float32:
    """`sqrt(n_fit_rows - 1)` forward, `1 / sqrt(n_fit_rows - 1)` inverse. `pca_validate` already refuses `n_rows <= 1` so the guard cannot fire on a fitted model; it is kept because dropping a guard that the reference wrote is a silent change of behavior on the one input it was written for."""
    var d = Float64(n_fit_rows - 1)
    if d <= 0.0:
        return Float32(0.0)
    var r = sqrt(d)
    if inverse:
        return Float32(1.0 / r)
    return Float32(r)


def whiten_scale_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    src: MutPointer[Float32, MutAnyOrigin],
    singular: MutPointer[Float32, MutAnyOrigin],
    n_cells_in: Int32,
    n_cols_in: Int32,
    scalar_in: Float32,
    divide_in: Int32,
):
    """`dst = whiten(src)`: cuML's two passes over the components copy, fused."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_cells_in):
        var c = i // Int(n_cols_in)
        var s = singular.unsafe_load(c)
        var v = ftz(identical_mul(src.unsafe_load(i), scalar_in))
        if abs(s) < Float32(WHITEN_SKIP_ZERO):
            dst.unsafe_store(i, v)
        elif divide_in != Int32(0):
            dst.unsafe_store(i, ftz(identical_div(v, s)))
        else:
            dst.unsafe_store(i, ftz(identical_mul(v, s)))


def whiten_components(
    ctx: DeviceContext,
    mut src: DeviceBuffer[DType.float32],
    mut dst: DeviceBuffer[DType.float32],
    mut singular: DeviceBuffer[DType.float32],
    n_components: Int,
    n_cols: Int,
    n_fit_rows: Int,
    inverse: Bool,
) raises:
    """Launch `whiten_scale_kernel` over the whole components matrix. `dst` is a separate buffer because cuML makes a copy too (`rmm::device_uvector<math_t> components_copy` at `pca.cuh:229` and `:289`): the caller's `components_` must survive the transform unchanged, the same contract `check_input_restored` holds the fit to."""
    var cells = n_components * n_cols
    var scalar = whiten_scalar(n_fit_rows, inverse)
    var divide = Int32(1)
    if inverse:
        divide = Int32(0)
    ctx.enqueue_function[whiten_scale_kernel](
        dst.unsafe_ptr(),
        src.unsafe_ptr(),
        singular.unsafe_ptr(),
        Int32(cells),
        Int32(n_cols),
        scalar,
        divide,
        grid_dim=((cells + 255) // 256, 1, 1),
        block_dim=(256, 1, 1),
    )
    ctx.synchronize()
