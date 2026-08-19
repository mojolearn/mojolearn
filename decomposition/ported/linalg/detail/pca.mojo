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

That flip is `sign_flip_kernel` below, a port of their `signFlipKernel`
(`raft/matrix/detail/math.cuh:358`), and it runs ON THE DEVICE where theirs
does. It replaced a host loop over the copied-back eigenvectors.
"""

from std.gpu import block_idx, thread_idx
from std.math import sqrt
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.primitives.block import max as block_max
from max.gpu.primitives.block import min as block_min
from max.gpu.sync import barrier
from std.memory import stack_allocation

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
    mut x_alias2: DeviceBuffer[DType.float32],
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
    gemm_tn(ctx, cov, x, x_alias, x_alias2, n_cols, n_cols, n_rows)
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
    mut x_alias2: DeviceBuffer[DType.float32],
    mut mu: DeviceBuffer[DType.float32],
    mut cov: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_cols: Int,
    n_components: Int,
    restore_input: Bool = True,
) raises -> PCAResult:
    """`pca_fit`, all six steps.

    The eigen half runs ON THE DEVICE, as cuSOLVER's `syevj` does for them:
    `jacobi_eigh_kernel` then `sign_flip_kernel`, both on the
    `n_cols x n_cols` matrix. What still comes back to the host is the
    ordering and the truncation; see `eig_and_truncate`.
    `decomposition/mojo_only/jacobi_eigh.mojo` is the host reference the
    device solver is checked against, not the path a fit takes.
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
    compute_covariance(
        ctx, x, x_alias, x_alias2, mu, cov, n_rows, n_cols, restore_input
    )

    return eig_and_truncate(
        ctx, cov, n_cols, n_components, n_rows - 1
    )


# RAFT dispatches `TPB` on the column LENGTH: 32 for `D <= 32`, then 64, 128,
# 256 (`math.cuh:391-400`). `jacobi_eigh_kernel` caps `n_cols` at
# `JACOBI_MAX_N = 32`, so the only arm of that dispatch this port can reach is
# the first one. The strided loops below are still written for any `n`.
comptime SIGNFLIP_TPB = 32


def sign_flip_kernel(
    v: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """PORT OF `signFlipKernel`, `raft/matrix/detail/math.cuh:358`.

    One block per component. Finds the entry of largest ABSOLUTE value in the
    component and negates the whole component if that entry is negative, which
    is the convention `sign_flip_components` fixes (`detail/pca.cuh:189` ->
    `detail/tsvd.cuh:173`) and the one scikit-learn's `svd_flip` uses.

    This REPLACED a host loop in `eig_and_truncate` that did the same thing on
    the copied-back eigenvectors. Same convention, same tie-break, on the
    device where theirs is.

    LAYOUT: theirs is column major, so a column is `D` CONTIGUOUS elements at
    `blockIdx.x * D`. `jacobi_eigh_kernel` writes eigenvector `c` into COLUMN
    `c` of a ROW MAJOR `n x n` matrix, so ours is the same column with a
    stride: entry `f` of component `c` is at `f * n + c`. That is the only
    change to their kernel.

    WHY TWO PASSES AND NOT `cub::BlockReduce<cub::KeyValuePair<int, T>>`.
    Theirs reduces a (index, value) pair with `cub::ArgMax`.
    `max.gpu.primitives.block.max` reduces VALUES only, and
    VENDOR_LIBRARIES.md records no block reduce over a custom operator or a
    pair type, so the argmax is split: `block_max` over `|entry|`, then
    `block_min` over the indices that attain it. The alternative was a
    warp-shuffle reduction carrying the index alongside the value, which
    `std.gpu.primitives.warp` can now express. Two passes won because the
    block collective is correct for any `block_size` without a cross-warp
    stage of our own, and because 32 columns of at most 32 entries is not a
    place to spend a hand-rolled reduction. The extra pass is O(n) on a
    matrix whose eigendecomposition already cost O(n^3).

    TIE-BREAK, WHICH IS PART OF THE CONVENTION. `cub::ArgMax` keeps the LOWER
    index on a tie, their own `thrust` variant of this loop keeps the first
    strict improvement (`detail/tsvd.cuh:290-297`, `if (val > max)`), and the
    host code this replaced did the same with `abs(v) > abs(biggest)`. Taking
    the MINIMUM index among the entries attaining the maximum reproduces all
    three. Without it, a component holding both `+x` and `-x` at the largest
    magnitude would have no defined sign, which is exactly the
    irreproducibility the sign flip exists to remove.

    NOTE ON WHAT CURRENT RAFT CALLS: `sign_flip_components`, the arm
    `pca_fit` actually reaches today, expresses this as a
    `raft::linalg::reduce` with a max-by-absolute-value custom operator
    (`detail/tsvd.cuh:229-243`). That is a device-wide reduce with a custom
    operator, which VENDOR_LIBRARIES.md records as NOT FOUND. `signFlipKernel`
    is the same answer written as an ordinary portable kernel, so it is the
    one that ports.
    """
    var n = Int(n_in)
    var col = Int(block_idx.x)
    var tid = Int(thread_idx.x)

    # `sh[0]` the largest magnitude, `sh[1]` the index holding it, `sh[2]` the
    # sign. Three slots and no reuse, so no read of one pass can race a write
    # of the next. Their `__shared__ bool need_sign_flip` is `sh[2]`.
    var sh = stack_allocation[
        3,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()

    # PASS 1: the largest magnitude in this component.
    var local = Float32(0.0)
    var f = tid
    while f < n:
        var m = abs(v.unsafe_load(f * n + col))
        if m > local:
            local = m
        f += SIGNFLIP_TPB
    var reduced_max = block_max[block_size=SIGNFLIP_TPB](local)
    # Only thread 0 is promised the reduction's result by CUB's contract, so
    # it is published through shared memory rather than assumed broadcast.
    if tid == 0:
        sh[0] = reduced_max
    barrier()
    var biggest = sh[0]

    # PASS 2: the LOWEST index attaining it. `n` is the sentinel for a thread
    # whose slice holds none. The equality is exact: a max is a selection, so
    # `biggest` is bit for bit one of the values compared against it.
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

    # PASS 3: the thread owning that index reads its SIGN. Exactly one thread
    # matches, because `biggest` is attained by at least one real entry.
    f = tid
    while f < n:
        if Float32(f) == first:
            if v.unsafe_load(f * n + col) < Float32(0.0):
                sh[2] = Float32(-1.0)
            else:
                sh[2] = Float32(1.0)
        f += SIGNFLIP_TPB
    barrier()

    # `if (need_sign_flip)`, `math.cuh:378`.
    if sh[2] < Float32(0.0):
        f = tid
        while f < n:
            v.unsafe_store(f * n + col, -v.unsafe_load(f * n + col))
            f += SIGNFLIP_TPB


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

    # `sign_flip_components`, `detail/pca.cuh:189`, ported as
    # `signFlipKernel` (`raft/matrix/detail/math.cuh:358`). One block per
    # eigenvector, and the whole `n_cols x n_cols` basis is fixed in one
    # launch before anything is copied back, which is what REPLACED the host
    # loop that used to run per SELECTED component further down. The
    # convention is unchanged: largest-magnitude entry positive.
    #
    # It runs over every column rather than only the `n_components` kept,
    # because the sign of a column does not depend on the ordering and doing
    # it here means the host side never touches signs at all. Theirs flips
    # `n_components` columns for the same reason: the flip is per column.
    ctx.enqueue_function[sign_flip_kernel](
        vec_buf.unsafe_ptr(),
        Int32(n_cols),
        grid_dim=(n_cols, 1, 1),
        block_dim=(SIGNFLIP_TPB, 1, 1),
    )
    ctx.synchronize()

    # Only the O(n_cols^2) post-processing comes back: ordering, truncation
    # and the variance ratios. RAFT does that part on device too
    # (`col_reverse`, `trunc_zero_origin`, `matrix::ratio`), so this is still
    # a departure and it is named in UNWIRED.md rather than glossed. It is
    # O(cols^2), never O(rows).
    #
    # The SIGN convention is no longer in that list: `sign_flip_kernel` above
    # applied it on the device, so `h_vec` arrives already flipped.
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

    # `cal_eig` + `col_reverse`: order DESCENDING by eigenvalue.
    #
    # A SORT, WHERE RAFT ONLY REVERSES, AND WHY THAT IS NOT A REGRESSION.
    # `cal_eig` calls cuSOLVER `syevj`, which returns eigenvalues ALREADY
    # ASCENDING, so `tsvd.cuh:149` reverses with `raft::matrix::col_reverse`
    # and `tsvd.cuh:156` reverses the variances with `row_reverse`: O(n) on
    # the device. **Cyclic Jacobi does not order anything.** Its output sits
    # in whatever order the rotations left it, so a reverse here would return
    # an arbitrary permutation, which for PCA is a DIFFERENT answer and not a
    # worse one. The sort is not a slower way to do their reverse, it is the
    # step their eigensolver already did for them.
    #
    # Left as an O(n_cols^2) selection sort on purpose. `jacobi_eigh_kernel`
    # caps `n_cols` at `JACOBI_MAX_N = 32` (one block, two `32 x 32` float
    # arrays in shared memory), so this is at most 496 comparisons and cannot
    # grow while that cap holds. It is a permutation of INDICES: no
    # arithmetic, so nothing here can move a number.
    #
    # WHAT TO DO WHEN THE CAP LIFTS. When the eigensolver takes an `n_cols`
    # that a single block cannot hold, this loop is the wrong shape and
    # `nn.argsort.argsort`, recorded AVAILABLE and device-capable in
    # VENDOR_LIBRARIES.md, is the replacement: sort the diagonal descending
    # on the device and gather. Not done now because it would be a device
    # call and a gather kernel added for a 32-element list, and because the
    # cap is the thing that has to move first.
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

        # The sign convention is already applied: `sign_flip_kernel` fixed
        # every column of `vec_buf` on the device before the copy back, so
        # the host only reads. This is where a host `abs()` loop per selected
        # component used to be.
        for f in range(n_cols):
            components.append(vecs[f * n_cols + src])
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
