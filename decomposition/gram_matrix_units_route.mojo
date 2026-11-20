# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of Mojolearn, https://doi.org/10.5281/zenodo.22068632
"""PCA AND TRUNCATED SVD ON THE MATRIX UNIT: THE DECOMPOSITION ROUTE OF `fast`
(lane fx-fast-nvidia-routes, 2026-11-19).

EXPERIMENT `-D MOJOLEARN_DECOMP_FAST_MATRIX_UNITS`, default OFF, A/B owed with quality beside time
(docs/apple-fast/EXPERIMENTS.md). Unverified: written without a compile. A family route of the matrix-units row of
checks/mode_policy.mojo (`ROUTE_DECOMP_MATRIX_UNITS`): legal only in a `fast` build (`-D MOJOLEARN_MODE_FAST_ROWS`);
a compile error in a build that promises identical bits. With the define off nothing here is compiled into a launch.

WHAT `identical` PAYS HERE. The Gram X^T X (PCA's covariance, truncated SVD's first step) and the projections
X V^T / scores V run one FMA chain per cell in a fixed order on the exact FP32 kernels (IDENTITY_PATHS rows 27, 28,
40). The Gram is m m k multiply-adds and is the fit; at widths past the split-K kernel it runs at the scalar
kernel's rate.

WHAT THE ROUTE DOES. Those products run on MAX `linalg.matmul`: on NVIDIA the tensor cores with TF32 operands and
FP32 accumulation, on AMD the fp32 MFMA (full fp32 operands; only the fold order is the vendor's). One source for
both. On Apple the PCA Gram already has its matrix-unit route (the `afn_launch_tile` arm of
decomposition/impl/linalg/detail/pca.mojo), so the row points at that and this route is off there; the host column
is exact.
  - Gram: `transpose_kernel` writes X^T row-major into a scratch of m k floats (data movement only; `linalg.matmul`
    documents `transpose_a` as unsupported), then Z = Xt . Xt^T. One allocation and one wait per fit, the same
    footprint the pre-identical route used (its two n d alias buffers).
  - Projection: Z[m x n] = X[m x k] . Y[n x k]^T, one launch, no workspace, asynchronous like the route it replaces.

WHAT STAYS EXACT. The split-K Gram (widths the split-K kernel holds: the product is small there and the kernel
fills the machine by splitting k, core/gram_splitk.mojo), every product under the size rule, the column means and
the centering, the eigensolver, the sign rule, the explained-variance arithmetic, and everything in OLS and ridge:
their Gram feeds a solve whose error is the operand error times the condition number of X^T X, so they are not
this route.

THE SIZE RULE: `vendor_matmul_size_rule` (gemm/fast_matrix_units.mojo: K >= 32, both output edges >= 64,
M N K >= 2^27; sizes and the hardware unit only) and, for the Gram, not a split-K shape.

QUALITY. NVIDIA: each operand keeps 11 significant bits, so a cell differs from its FP32 value by at most
`matrix_units_cell_bound` (1.95e-3 of the cell's ABSOLUTE sum; typically 4e-4 / sqrt(K) of it). For a centered
Gram the diagonal is its own absolute sum, so each variance moves by at most 0.2 percent and the trace
(total variance) by the same; an eigenvalue moves by at most the perturbation's norm (Weyl), so components whose
eigenvalue is under about 1e-3 of the largest are no longer resolved from each other. The check
(checks/native_tests/decomp_matrix_units.mojo) holds every Gram and projection cell to the bound; the owed A/B
compares explained_variance_ratio, the reconstruction error and the projection against the zero-row fast set.
"""
from layout import TileTensor
from layout.tile_layout import row_major
from linalg.matmul import matmul
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.mode_policy import ROUTE_DECOMP_MATRIX_UNITS
from core.column_stats import CUDA_MAX_GRID_YZ, TRANSPOSE_TILE, transpose_kernel
from core.gram_splitk import gram_splitk_applies
from gemm.fast_matrix_units import vendor_matmul_size_rule

#: The route: PCA / truncated-SVD products on the matrix unit in an NVIDIA or AMD `fast` build.
comptime DECOMP_MATRIX_UNITS = ROUTE_DECOMP_MATRIX_UNITS


def decomp_gram_units_takes(m: Int, k: Int) raises -> Bool:
    """Whether the route takes the Gram of a k x m matrix: past the split-K kernel and inside the size rule.
    Always False without the route."""
    comptime if not DECOMP_MATRIX_UNITS:
        return False
    if gram_splitk_applies(m, m, k):
        return False
    return vendor_matmul_size_rule(m, m, k)


def decomp_gram_units_into(
    ctx: DeviceContext,
    mut z: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    m: Int,
    k: Int,
) raises -> Bool:
    """`z[m x m] = x[k x m]^T . x[k x m]` on the matrix unit. True if it took the product (z is written and
    the queue is drained, as after `gemm_tn`'s own allocating branch); False leaves every buffer untouched and
    the caller runs `gemm_tn`. The caller has validated the operand lengths."""
    comptime if not DECOMP_MATRIX_UNITS:
        return False
    if not decomp_gram_units_takes(m, k):
        return False
    if len(z) < m * m or len(x) < k * m:
        return False
    var xt = ctx.enqueue_create_buffer[DType.float32](m * k)
    # xt[m x k] = x[k x m]^T: the launch of core/gemm.mojo::gemm_tn_via_transpose
    ctx.enqueue_function[transpose_kernel](
        xt.unsafe_ptr(),
        x.unsafe_ptr(),
        Int32(k),
        Int32(m),
        grid_dim=(
            (m + TRANSPOSE_TILE - 1) // TRANSPOSE_TILE,
            min((k + TRANSPOSE_TILE - 1) // TRANSPOSE_TILE, CUDA_MAX_GRID_YZ),
            1,
        ),
        block_dim=(TRANSPOSE_TILE, TRANSPOSE_TILE, 1),
    )
    # The self-Gram reads `xt` as both operands (the form of core/gemm.mojo::gemm_nt_gram). `matmul` takes its
    # operands mutably, and one variable may not be passed mutably twice ("aliasing values passed mutably to 'a'
    # and 'b'"), so the second operand is a second HANDLE of the same device storage: a reference-count copy of
    # the buffer handle, no second copy of X^T and no extra device memory.
    var xt_b = xt
    var tz = TileTensor(z, row_major(m, m))
    var tx = TileTensor(xt, row_major(m, k))
    var ty = TileTensor(xt_b, row_major(m, k))
    matmul[transpose_b=True, target="gpu"](tz, tx, ty, ctx)
    # `xt` is this function's: wait for the two launches that read it before it is released.
    ctx.synchronize()
    _ = xt_b^
    _ = xt^
    return True


def decomp_project_units_into(
    ctx: DeviceContext,
    mut z: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    mut y: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
) raises -> Bool:
    """`z[m x n] = x[m x k] . y[n x k]^T` on the matrix unit (a transform or an inverse transform). True if it
    took the product; False leaves every buffer untouched and the caller runs `gemm_nt`. Asynchronous: no
    wait, the caller keeps z, x and y alive past its own synchronize, as for `gemm_nt`."""
    comptime if not DECOMP_MATRIX_UNITS:
        return False
    if not vendor_matmul_size_rule(m, n, k):
        return False
    if len(z) < m * n or len(x) < m * k or len(y) < n * k:
        return False
    var tz = TileTensor(z, row_major(m, n))
    var tx = TileTensor(x, row_major(m, k))
    var ty = TileTensor(y, row_major(n, k))
    matmul[transpose_b=True, target="gpu"](tz, tx, ty, ctx)
    return True
