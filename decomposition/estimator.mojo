# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Host-pointer surfaces for PCA and truncated SVD.

THE IDENTITY CARD (DEVIATION 518, 2026-08-23 -- the same deviation also
repairs the k-means++ card, see `cluster/ported/cluster/detail/kmeans.mojo`
`init_scalable_kmeans_plus_plus`). `pca_fit_host` and `tsvd_fit_host` are
the paths `mojolearn.PCA` / `mojolearn.TruncatedSVD` take, and neither left
a stage card: `decomposition/` has no `IdentityTrace` anywhere below this
file, so `tools/e2u_matrix_fit.py` could hash the outputs and nothing else.
The records below are taken AT THIS SURFACE, from the buffers the ported
fit hands back:

    pca.mean              the column means (`mu`, device)
    pca.jacobi.a          the covariance AFTER the device Jacobi, in place;
                          its diagonal is the eigenvalues. A divergence
                          here with `pca.mean` agreeing is the Gram/covariance
                          or the eigensolver (DEVIATION 511's two block.sum
                          folds live there and are NOT pinned)
    pca.components        what the caller gets, after the host ordering,
                          truncation and sign convention
    pca.explained_var
    pca.singular_vals
    pca.noise_var

and `tsvd.jacobi.a` / `tsvd.components` / `tsvd.singular_vals` for the
uncentered twin. WHAT IS MISSING, named rather than glossed: a record of
the covariance BEFORE the Jacobi (`compute_covariance`'s output). It would
separate the Gram from the eigensolver and it needs one line inside
`decomposition/ported/linalg/detail/pca.mojo::pca_fit`, which is the
decomposition lane's file; left for that lane.
"""

from max.gpu.host import DeviceContext

from core.column_stats import TRANSPOSE_TILE, shift_columns_kernel, transpose_kernel
from core.identity_trace import IdentityTrace
from core.gemm import gemm_nt
from decomposition.ported.linalg.detail.pca import (
    compute_covariance,
    eig_and_truncate,
    pca_transform,
    pca_validate,
)
from core.gemm import gemm_tn


def pca_fit_host(
    ctx: DeviceContext,
    x_ptr: MutPointer[Float32, MutUntrackedOrigin],
    components_ptr: MutPointer[Float32, MutUntrackedOrigin],
    mean_ptr: MutPointer[Float32, MutUntrackedOrigin],
    explained_ptr: MutPointer[Float32, MutUntrackedOrigin],
    ratio_ptr: MutPointer[Float32, MutUntrackedOrigin],
    singular_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_rows: Int,
    n_features: Int,
    n_components: Int,
) raises -> Float64:
    var x = ctx.enqueue_create_buffer[DType.float32](n_rows * n_features)
    var xa = ctx.enqueue_create_buffer[DType.float32](n_rows * n_features)
    var xa2 = ctx.enqueue_create_buffer[DType.float32](n_rows * n_features)
    var mu = ctx.enqueue_create_buffer[DType.float32](n_features)
    var cov = ctx.enqueue_create_buffer[DType.float32](n_features * n_features)
    ctx.enqueue_copy(dst_buf=x, src_ptr=x_ptr)
    ctx.synchronize()
    var trace = IdentityTrace()
    if trace.enabled:
        trace.header(
            String("pca n=") + String(n_rows) + " d=" + String(n_features)
            + " n_components=" + String(n_components)
        )
    # The two halves of `pca_fit`, called here by name so the covariance
    # can be recorded BETWEEN them (IDENTITY_PATHS row 38, 2026-08-23): the
    # Jacobi kernel diagonalizes `cov` IN PLACE, so a card that records it
    # only after the fit holds the eigensolver's output and cannot say
    # whether a divergence began in the Gram (rows 27/29) or in the solve
    # (row 31). `pca.cov` is the product as the solver receives it;
    # `pca.jacobi.a` keeps its name and is the matrix the solver left.
    pca_validate(n_rows, n_features, n_components)
    compute_covariance(ctx, x, xa, xa2, mu, cov, n_rows, n_features, True)
    if trace.enabled:
        trace.record_device(ctx, "pca.mean", mu, n_features)
        trace.record_device(ctx, "pca.cov", cov, n_features * n_features)
    var result = eig_and_truncate(
        ctx, cov, n_features, n_components, n_rows - 1
    )
    if trace.enabled:
        trace.record_device(ctx, "pca.jacobi.a", cov, n_features * n_features)
    var comp32 = List[Float32]()
    var expl32 = List[Float32]()
    var sing32 = List[Float32]()
    for i in range(n_components * n_features):
        comp32.append(Float32(result.components[i]))
        components_ptr.unsafe_store(i, Float32(result.components[i]))
    for i in range(n_components):
        expl32.append(Float32(result.explained_var[i]))
        sing32.append(Float32(result.singular_vals[i]))
        explained_ptr.unsafe_store(i, Float32(result.explained_var[i]))
        ratio_ptr.unsafe_store(i, Float32(result.explained_var_ratio[i]))
        singular_ptr.unsafe_store(i, Float32(result.singular_vals[i]))
    if trace.enabled:
        trace.record_list_f32("pca.components", comp32)
        trace.record_list_f32("pca.explained_var", expl32)
        trace.record_list_f32("pca.singular_vals", sing32)
        trace.record_scalar_f32("pca.noise_var", Float32(result.noise_var))
    var hmu = ctx.enqueue_create_host_buffer[DType.float32](n_features)
    ctx.enqueue_copy(dst_ptr=hmu.unsafe_ptr(), src_buf=mu)
    ctx.synchronize()
    for i in range(n_features):
        mean_ptr.unsafe_store(i, hmu.unsafe_ptr().unsafe_load(i))
    return result.noise_var


def pca_transform_host(
    ctx: DeviceContext,
    x_ptr: MutPointer[Float32, MutUntrackedOrigin],
    mean_ptr: MutPointer[Float32, MutUntrackedOrigin],
    components_ptr: MutPointer[Float32, MutUntrackedOrigin],
    out_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_rows: Int,
    n_features: Int,
    n_components: Int,
) raises:
    var x = ctx.enqueue_create_buffer[DType.float32](n_rows * n_features)
    var mu = ctx.enqueue_create_buffer[DType.float32](n_features)
    var components = ctx.enqueue_create_buffer[DType.float32](n_components * n_features)
    var out = ctx.enqueue_create_buffer[DType.float32](n_rows * n_components)
    ctx.enqueue_copy(dst_buf=x, src_ptr=x_ptr)
    ctx.enqueue_copy(dst_buf=mu, src_ptr=mean_ptr)
    ctx.enqueue_copy(dst_buf=components, src_ptr=components_ptr)
    ctx.synchronize()
    pca_transform(ctx, x, mu, components, out, n_rows, n_features, n_components)
    ctx.enqueue_copy(dst_ptr=out_ptr, src_buf=out)
    ctx.synchronize()


def tsvd_fit_host(
    ctx: DeviceContext,
    x_ptr: MutPointer[Float32, MutUntrackedOrigin],
    components_ptr: MutPointer[Float32, MutUntrackedOrigin],
    singular_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_rows: Int,
    n_features: Int,
    n_components: Int,
) raises:
    var x = ctx.enqueue_create_buffer[DType.float32](n_rows * n_features)
    var gram = ctx.enqueue_create_buffer[DType.float32](n_features * n_features)
    var xa = ctx.enqueue_create_buffer[DType.float32](n_rows * n_features)
    var xa2 = ctx.enqueue_create_buffer[DType.float32](n_rows * n_features)
    ctx.enqueue_copy(dst_buf=x, src_ptr=x_ptr)
    ctx.synchronize()
    var trace = IdentityTrace()
    if trace.enabled:
        trace.header(
            String("tsvd n=") + String(n_rows) + " d=" + String(n_features)
            + " n_components=" + String(n_components)
        )
    # The two halves of `tsvd_fit` by name, the Gram recorded between them
    # (row 38; see the PCA surface above for why).
    pca_validate(n_rows, n_features, n_components)
    gemm_tn(ctx, gram, x, xa, xa2, n_features, n_features, n_rows)
    ctx.synchronize()
    if trace.enabled:
        trace.record_device(ctx, "tsvd.gram", gram, n_features * n_features)
    var result = eig_and_truncate(ctx, gram, n_features, n_components, 1)
    if trace.enabled:
        trace.record_device(
            ctx, "tsvd.jacobi.a", gram, n_features * n_features
        )
    var comp32 = List[Float32]()
    var sing32 = List[Float32]()
    for i in range(n_components * n_features):
        comp32.append(Float32(result.components[i]))
        components_ptr.unsafe_store(i, Float32(result.components[i]))
    for i in range(n_components):
        sing32.append(Float32(result.singular_vals[i]))
        singular_ptr.unsafe_store(i, Float32(result.singular_vals[i]))
    if trace.enabled:
        trace.record_list_f32("tsvd.components", comp32)
        trace.record_list_f32("tsvd.singular_vals", sing32)


def tsvd_transform_host(
    ctx: DeviceContext,
    x_ptr: MutPointer[Float32, MutUntrackedOrigin],
    components_ptr: MutPointer[Float32, MutUntrackedOrigin],
    out_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_rows: Int,
    n_features: Int,
    n_components: Int,
) raises:
    var x = ctx.enqueue_create_buffer[DType.float32](n_rows * n_features)
    var components = ctx.enqueue_create_buffer[DType.float32](n_components * n_features)
    var out = ctx.enqueue_create_buffer[DType.float32](n_rows * n_components)
    ctx.enqueue_copy(dst_buf=x, src_ptr=x_ptr)
    ctx.enqueue_copy(dst_buf=components, src_ptr=components_ptr)
    ctx.synchronize()
    gemm_nt(ctx, out, x, components, n_rows, n_components, n_features)
    ctx.enqueue_copy(dst_ptr=out_ptr, src_buf=out)
    ctx.synchronize()


def inverse_transform_host(
    ctx: DeviceContext,
    scores_ptr: MutPointer[Float32, MutUntrackedOrigin],
    components_ptr: MutPointer[Float32, MutUntrackedOrigin],
    mean_ptr: MutPointer[Float32, MutUntrackedOrigin],
    out_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_rows: Int,
    n_features: Int,
    n_components: Int,
    add_mean: Bool,
) raises:
    var scores = ctx.enqueue_create_buffer[DType.float32](n_rows * n_components)
    var components = ctx.enqueue_create_buffer[DType.float32](n_components * n_features)
    var components_t = ctx.enqueue_create_buffer[DType.float32](n_features * n_components)
    var mean = ctx.enqueue_create_buffer[DType.float32](n_features)
    var out = ctx.enqueue_create_buffer[DType.float32](n_rows * n_features)
    ctx.enqueue_copy(dst_buf=scores, src_ptr=scores_ptr)
    ctx.enqueue_copy(dst_buf=components, src_ptr=components_ptr)
    if add_mean:
        ctx.enqueue_copy(dst_buf=mean, src_ptr=mean_ptr)
    ctx.enqueue_function[transpose_kernel](
        components_t.unsafe_ptr(), components.unsafe_ptr(),
        Int32(n_components), Int32(n_features),
        grid_dim=((n_features + TRANSPOSE_TILE - 1) // TRANSPOSE_TILE,
                  (n_components + TRANSPOSE_TILE - 1) // TRANSPOSE_TILE, 1),
        block_dim=(TRANSPOSE_TILE, TRANSPOSE_TILE, 1),
    )
    ctx.synchronize()
    gemm_nt(ctx, out, scores, components_t, n_rows, n_features, n_components)
    if add_mean:
        ctx.enqueue_function[shift_columns_kernel](
            out.unsafe_ptr(), mean.unsafe_ptr(), Int32(n_rows), Int32(n_features),
            Float32(1.0), grid_dim=((n_rows * n_features + 255) // 256, 1, 1),
            block_dim=(256, 1, 1),
        )
    ctx.enqueue_copy(dst_ptr=out_ptr, src_buf=out)
    ctx.synchronize()
