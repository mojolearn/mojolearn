"""PCA by covariance eigendecomposition.

PORT OF `raft/linalg/detail/pca.cuh` and the `cal_eig` /
`trunc_comp_exp_vars` pair from `raft/linalg/detail/tsvd.cuh` at RAFT
`9aa17e5`. Partial. Do not improve.

**RAFT's PCA is NOT randomized SVD**, which is what I expected to find and is
worth stating because it changes what the port costs. `pca_fit` is six steps
(`detail/pca.cuh:128-190`):

    1  mean over columns                     -> mu           O(rows)
    2  center the input IN PLACE             ->              O(rows)
    3  covariance, Xc^T Xc / (n_rows - 1)    -> cov          O(rows * cols^2)
    4  eigendecompose cov, take the top k    -> components   O(cols^3)
    5  singular_vals = sqrt(var * (n-1))     ->              O(k)
    6  RESTORE the input by adding mu back   ->              O(rows)

**Only steps 1, 2, 3 and 6 touch rows at all.** Everything after step 3 works
on an `n_cols x n_cols` matrix. That is the structural reason this algorithm
suits a GPU so well: one bandwidth-bound pass and one big arithmetic-dense
product, then a small dense problem that does not care where it runs.

Step 6 is the one to not drop. `input` is an in-out parameter that must end
the call unchanged, and a fit that leaves the caller's matrix centered is
wrong in a way nothing in the fit itself will reveal.

ORDER IS PART OF THE ANSWER
---------------------------
`cal_eig` gets ascending eigenvalues from cuSOLVER and then calls
`raft::matrix::col_reverse` (`tsvd.cuh:150`) to make components descend by
explained variance. Copied. PCA components in the wrong order are not a
lesser answer, they are a different one.

SIGN IS ALSO PART OF THE ANSWER
-------------------------------
An eigenvector is defined up to sign, so any implementation must PICK one or
its output is not reproducible. RAFT calls `sign_flip_components` with a
`flip_signs_based_on_U` switch (`detail/pca.cuh:189`). This ports the default
arm, which fixes the sign from the components themselves: make the entry of
largest magnitude in each component positive. scikit-learn's `svd_flip` uses
the same convention, which is what makes the two comparable at all.
"""

from std.math import sqrt
from max.gpu.host import DeviceBuffer, DeviceContext

from cluster.mojo_only.reduce_by_key import copy_f32_kernel
from core.gemm import gemm_nt, gemm_tn
from core.column_stats import (
    COV_TILE,
    STATS_TPB,
    column_mean_kernel,
    scale_in_place_kernel,
    shift_columns_kernel,
)
from decomposition.mojo_only.jacobi_eigh import jacobi_eigh
from decomposition.mojo_only.jacobi_eigh_device import (
    JACOBI_TPB,
    jacobi_eigh_kernel,
)


@fieldwise_init
struct PCAResult(Movable):
    """What `pca_fit` writes back on the host side.

    `components` is `n_components x n_cols` row major, matching theirs and
    scikit-learn's `components_`.
    """

    var components: List[Float64]
    var explained_var: List[Float64]
    var explained_var_ratio: List[Float64]
    var singular_vals: List[Float64]
    var noise_var: Float64


def compute_covariance(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    mut x_alias: DeviceBuffer[DType.float32],
    mut mu: DeviceBuffer[DType.float32],
    mut cov: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_cols: Int,
    restore_input: Bool = True,
) raises:
    """Steps 1, 2, 3 and 6. The only part of PCA that scales with rows."""
    ctx.enqueue_function[column_mean_kernel](
        mu.unsafe_ptr(),
        x.unsafe_ptr(),
        Int32(n_rows),
        Int32(n_cols),
        grid_dim=(n_cols, 1, 1),
        block_dim=(STATS_TPB, 1, 1),
    )
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
    # `raft::stats::cov` is a GEMM with `CUBLAS_OP_T, CUBLAS_OP_N`, so it goes
    # through the same tuned matmul as everything else. This was the SECOND
    # contraction-shaped kernel in `core/` and it is why PCA did not move for
    # four benchmark rounds while the other one was being tuned.
    #
    # MAX's matmul has no `alpha`, so cuBLAS's scale becomes its own pass over
    # `n_cols^2` elements. A deviation in launch count, not arithmetic.
    gemm_tn(ctx, cov, x, x_alias, n_cols, n_cols, n_rows)
    ctx.enqueue_function[scale_in_place_kernel](
        cov.unsafe_ptr(),
        Int32(n_cols * n_cols),
        Float32(1.0) / Float32(n_rows - 1),
        grid_dim=((n_cols * n_cols + 255) // 256, 1, 1),
        block_dim=(256, 1, 1),
    )
    if restore_input:
        # `raft::stats::meanAdd`, `detail/pca.cuh:186`. Do not drop this.
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


def pca_fit(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    mut x_alias: DeviceBuffer[DType.float32],
    mut mu: DeviceBuffer[DType.float32],
    mut cov: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_cols: Int,
    n_components: Int,
    restore_input: Bool = True,
) raises -> PCAResult:
    """`pca_fit`, all six steps.

    The eigen half runs on the host on an `n_cols x n_cols` matrix; see
    `decomposition/mojo_only/jacobi_eigh.mojo` for why that is inside
    `HOST_AND_DEVICE.md`'s rule and not an exception to it.
    """
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

    # Mojo refuses one buffer as two mutable kernel arguments (PORTING.md 24),
    # and X^T X names X twice, so the caller supplies an aliased copy.
    ctx.enqueue_function[copy_f32_kernel](
        x_alias.unsafe_ptr(),
        x.unsafe_ptr(),
        Int32(n_rows * n_cols),
        grid_dim=((n_rows * n_cols + 255) // 256, 1, 1),
        block_dim=(256, 1, 1),
    )
    compute_covariance(ctx, x, x_alias, mu, cov, n_rows, n_cols, restore_input)

    return eig_and_truncate(
        ctx, cov, n_cols, n_components, n_rows - 1
    )


def eig_and_truncate(
    ctx: DeviceContext,
    mut cov: DeviceBuffer[DType.float32],
    n_cols: Int,
    n_components: Int,
    singular_scale: Int,
) raises -> PCAResult:
    """`cal_eig` + `trunc_comp_exp_vars`, shared by PCA and truncated SVD.

    Both callers reach this with an `n_cols x n_cols` symmetric matrix and
    differ only in how they built it: PCA from CENTERED data scaled by
    `1 / (n_rows - 1)`, truncated SVD from RAW data with no scaling. Their
    two files call the same `cal_eig`, so this is one function here too.

    `singular_scale` is the `n_rows - 1` factor `pca_fit` multiplies by
    before taking the square root (`detail/pca.cuh:180`). `tsvd_fit` passes
    1, because its eigenvalues are already the squared singular values.
    """
    # `cal_eig` -> cuSOLVER `syevj`, WHICH RUNS ON THE DEVICE. So this runs
    # on the device too. The first version of this port put it on the host,
    # which was inside HOST_AND_DEVICE.md's O(rows) rule but was NOT a mirror
    # of their host/device split, and mirroring that split is the standing
    # rule. `jacobi_eigh.mojo` survives as the reference this is checked
    # against.
    var vec_buf = ctx.enqueue_create_buffer[DType.float32](n_cols * n_cols)
    ctx.synchronize()
    ctx.enqueue_function[jacobi_eigh_kernel](
        cov.unsafe_ptr(),
        vec_buf.unsafe_ptr(),
        Int32(n_cols),
        Int32(80),
        Float32(1.0e-10),
        grid_dim=(1, 1, 1),
        block_dim=(JACOBI_TPB, 1, 1),
    )
    ctx.synchronize()

    # Only the O(n_cols^2) post-processing comes back: ordering, truncation,
    # the sign convention and the variance ratios. RAFT does that part on
    # device too (`col_reverse`, `trunc_zero_origin`, `matrix::ratio`), so
    # this is still a departure and it is named in UNWIRED.md rather than
    # glossed. It is O(cols^2), never O(rows).
    var h_cov = ctx.enqueue_create_host_buffer[DType.float32](n_cols * n_cols)
    var h_vec = ctx.enqueue_create_host_buffer[DType.float32](n_cols * n_cols)
    ctx.enqueue_copy(dst_ptr=h_cov.unsafe_ptr(), src_buf=cov)
    ctx.enqueue_copy(dst_ptr=h_vec.unsafe_ptr(), src_buf=vec_buf)
    ctx.synchronize()

    var a = List[Float64]()
    for i in range(n_cols * n_cols):
        a.append(Float64(h_cov.unsafe_ptr().unsafe_load(i)))
    var vecs = List[Float64]()
    for i in range(n_cols * n_cols):
        vecs.append(Float64(h_vec.unsafe_ptr().unsafe_load(i)))

    # `cal_eig` + `col_reverse`: sort DESCENDING by eigenvalue.
    var order = List[Int]()
    for i in range(n_cols):
        order.append(i)
    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            if a[order[j] * n_cols + order[j]] > a[order[i] * n_cols + order[i]]:
                var t = order[i]
                order[i] = order[j]
                order[j] = t

    var total = 0.0
    for i in range(n_cols):
        total += a[i * n_cols + i]

    var components = List[Float64]()
    var explained_var = List[Float64]()
    var explained_var_ratio = List[Float64]()
    var singular_vals = List[Float64]()

    for c in range(n_components):
        var src = order[c]
        var lam = a[src * n_cols + src]

        # sign_flip: make the largest-magnitude entry positive.
        var biggest = 0.0
        var sign = 1.0
        for f in range(n_cols):
            var v = vecs[f * n_cols + src]
            if abs(v) > abs(biggest):
                biggest = v
        if biggest < 0.0:
            sign = -1.0

        for f in range(n_cols):
            components.append(sign * vecs[f * n_cols + src])
        explained_var.append(lam)
        explained_var_ratio.append(lam / total if total != 0.0 else 0.0)
        # `weighted_sqrt(explained_var, n_rows - 1)`, `detail/pca.cuh:180`.
        singular_vals.append(sqrt(lam * Float64(singular_scale)))

    # `noise_vars`: the mean of the DISCARDED eigenvalues, or zero if none
    # were discarded (`trunc_comp_exp_vars`, tail).
    var noise = 0.0
    if n_components < n_cols:
        for c in range(n_components, n_cols):
            noise += a[order[c] * n_cols + order[c]]
        noise /= Float64(n_cols - n_components)

    return PCAResult(
        components^, explained_var^, explained_var_ratio^, singular_vals^, noise
    )



    return PCAResult(
        components^, explained_var^, explained_var_ratio^, singular_vals^, noise
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
    """`pca_transform`: center, then project onto the components.

    The projection is `Xc . components^T`, which is exactly the shape
    `core/gemm.mojo` already computes, so the transform needs no new kernel.
    The input is centered and restored around it, same as the fit.
    """
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
