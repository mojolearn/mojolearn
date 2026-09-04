# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""PCA by covariance eigendecomposition. The `input`-unchanged CONTRACT is the one to not drop: `input` is an in-out parameter that must end the call unchanged, and a fit that leaves the caller's matrix centered is wrong in a way nothing in the fit itself will reveal."""

from std.gpu import block_dim, block_idx, thread_idx
from std.math import sqrt
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from core.pinned_reduce import (
    pinned_block_max as block_max,
    pinned_block_min as block_min,
)
from max.gpu.sync import barrier
from std.memory import stack_allocation

from core.gemm import gemm_nt, gemm_tn
from checks.numerics import ftz, identical_div, identical_mul
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
    JACOBI_TPB,
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
    """PORT OF `signFlipKernel`, `raft/matrix/detail/math.cuh:367`. `decomposition/checks/pca_check.mojo` holds it to that: the device answer must equal a fold-free host scan BITWISE, and the tie, the zero cases and the NaN case are each planted rather than hoped for."""
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
) raises -> PCAResult:
    """`colReverse` + `truncCompExpVars`: the HOST tail of a fit, shared."""

    var order = List[Int]()
    for i in range(n_cols):
        order.append(i)
    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            if diag[order[j]] > diag[order[i]]:
                var t = order[i]
                order[i] = order[j]
                order[j] = t

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
            components.append(vecs[f * n_cols + src])
        explained_var.append(lam)
        explained_var_ratio.append(lam / total if total != 0.0 else 0.0)
        singular_vals.append(sqrt(lam * Float64(singular_scale)))

    var noise = 0.0
    if n_components < n_cols and n_components <= singular_scale:
        for c in range(n_components, n_cols):
            noise += diag[order[c]]
        noise /= Float64(n_cols - n_components)

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
    ctx.enqueue_function[jacobi_eigh_kernel](
        cov.unsafe_ptr(),
        vec_buf.unsafe_ptr(),
        info_buf.unsafe_ptr(),
        Int32(n_cols),
        Int32(JACOBI_SWEEPS),
        Float32(JACOBI_TOL),
        grid_dim=(1, 1, 1),
        block_dim=(JACOBI_TPB, 1, 1),
    )

    ctx.enqueue_function[sign_flip_kernel](
        vec_buf.unsafe_ptr(),
        Int32(n_cols),
        grid_dim=(n_cols, 1, 1),
        block_dim=(SIGNFLIP_TPB, 1, 1),
    )
    ctx.synchronize()

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
    """`sqrt(n_fit_rows - 1)` forward, `1 / sqrt(n_fit_rows - 1)` inverse. `pca_validate` already refuses `n_rows <= 1` so the guard cannot fire on a fitted model; it is kept because dropping a guard that upstream wrote is a silent change of behavior on the one input it was written for."""
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
