# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Host-pointer surfaces for PCA and truncated SVD.

THE IDENTITY CARD (DEVIATION 518, 2026-08-23 -- the same deviation also
repairs the k-means++ card, see `cluster/impl/detail/kmeans.mojo`
`init_scalable_kmeans_plus_plus`). `pca_fit_host` and `tsvd_fit_host` are
the paths `mojolearn.PCA` / `mojolearn.TruncatedSVD` take, and neither left
a stage card: `decomposition/` has no `IdentityTrace` anywhere below this
file, so `tools/e2u_matrix_fit.py` could hash the outputs and nothing else.
The records below are taken AT THIS SURFACE, from the buffers the implemented
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
`decomposition/impl/linalg/detail/pca.mojo::pca_fit`, which is the
decomposition lane's file; left for that lane.
"""

from max.gpu.host import DeviceContext

from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer

from checks.numerics import ftz, identical_mul, GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from std.sys.compile import is_defined
from decomposition.tsvd_finish import TSVD_FIN_TPB, tsvd_slot_sum, tsvd_ratio
from x_linear.ff import FF, ff_add, ff_f32
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from core.column_stats import (
    STATS_TPB,
    TRANSPOSE_TILE,
    column_mean_kernel,
    shift_columns_kernel,
    transpose_kernel,
)
from core.identity_trace import IdentityTrace
from core.xtdz_coalesced import column_mean_launch
from core.gemm import gemm_nt
from decomposition.impl.linalg.detail.pca import (
    PCA_FAST_GRAM_MMA,
    compute_covariance,
    eig_and_truncate,
    pca_transform,
    pca_validate,
    whiten_components,
)
from core.gemm import gemm_tn
from decomposition.impl.linalg.detail.svd_full import (
    pca_fit_full,
    pca_full_scratch_cells,
)


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
    # PCA_FAST_GRAM_MMA: `x` is this fit's own device copy, never read again,
    # so its restore pass is skipped
    compute_covariance(ctx, x, xa, xa2, mu, cov, n_rows, n_features, not PCA_FAST_GRAM_MMA)
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


def pca_fit_full_host(
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
    """`PCA(svd_solver='full')`: the R-SVD arm, same five outputs.

    The twin of `pca_fit_host` and deliberately the same shape, so a caller
    switching solvers changes ONE name. What differs is inside: no
    covariance is formed, so the identity card records `pca.full.r` (the QR's
    factor, `n_features^2`) where the covariance arm records `pca.cov`, and
    `pca.full.singular` where it records `pca.jacobi.a`.

    THE CARD'S STAGES ARE THE DIAGNOSTIC. A divergence in `pca.mean` with
    `pca.full.r` agreeing is impossible; `pca.mean` agreeing and
    `pca.full.r` diverging is the QR (DEVIATIONS 586-589); both agreeing and
    `pca.full.singular` diverging is the one-sided Jacobi (DEVIATION 590);
    everything agreeing and `pca.full.components` diverging is the sign flip
    or the host ordering, which are SHARED with the covariance arm and would
    therefore diverge there too.
    """
    var x = ctx.enqueue_create_buffer[DType.float32](n_rows * n_features)
    var scratch = ctx.enqueue_create_buffer[DType.float32](
        pca_full_scratch_cells(n_rows, n_features)
    )
    var r = ctx.enqueue_create_buffer[DType.float32](n_features * n_features)
    var v = ctx.enqueue_create_buffer[DType.float32](n_features * n_features)
    var s = ctx.enqueue_create_buffer[DType.float32](n_features)
    var mu = ctx.enqueue_create_buffer[DType.float32](n_features)
    ctx.enqueue_copy(dst_buf=x, src_ptr=x_ptr)
    ctx.synchronize()
    var trace = IdentityTrace()
    if trace.enabled:
        trace.header(
            String("pca-full n=") + String(n_rows) + " d=" + String(n_features)
            + " n_components=" + String(n_components)
        )
    var result = pca_fit_full(
        ctx, x, scratch, r, v, s, mu, n_rows, n_features, n_components
    )
    if trace.enabled:
        trace.record_device(ctx, "pca.mean", mu, n_features)
        trace.record_device(ctx, "pca.full.r", r, n_features * n_features)
        trace.record_device(ctx, "pca.full.singular", s, n_features)
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
        trace.record_list_f32("pca.full.components", comp32)
        trace.record_list_f32("pca.full.explained_var", expl32)
        trace.record_list_f32("pca.full.singular_vals", sing32)
        trace.record_scalar_f32("pca.full.noise_var", Float32(result.noise_var))
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


def square_in_place_kernel(
    a: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """`a[i] = ftz(a[i] * a[i])`, the multiply pinned (lane/algos-decomp,
    2026-09-27; `tsvd_explained_host`)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        var v = a.unsafe_load(i)
        a.unsafe_store(i, ftz(identical_mul(v, v)))


# lane fam2-decomp (2026-10-04), IDENTICAL: the column mean is launched
# through core/xtdz_coalesced.mojo `column_mean_launch` (the launch
# decomposition/impl/linalg/detail/pca.mojo already uses): the same chains,
# fold and quotient as `column_mean_kernel`, read row-coalesced where that
# form applies. Same words. -D MOJOLEARN_IDN_DECOMP_MEAN_LAUNCH_OFF (or
# -D MOJOLEARN_IDN_ALL_OFF) restores the direct one-block-per-column launch.
comptime IDN_DECOMP_MEAN_LAUNCH = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not (is_defined["MOJOLEARN_IDN_DECOMP_MEAN_LAUNCH_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]())


def _column_variance(
    ctx: DeviceContext,
    mut m: DeviceBuffer[DType.float32],
    mut mu: DeviceBuffer[DType.float32],
    mut var_out: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_cols: Int,
) raises:
    """numpy's `var(axis=0)` (ddof 0) of `m`, IN PLACE over `m`: the column
    mean (`column_mean_kernel`), the centering (`shift_columns_kernel`),
    the pinned square, then the column mean of the squares."""
    var cells = n_rows * n_cols
    comptime if IDN_DECOMP_MEAN_LAUNCH:
        column_mean_launch(ctx, mu, m, n_rows, n_cols)
    else:
        ctx.enqueue_function[column_mean_kernel](
            mu.unsafe_ptr(), m.unsafe_ptr(), Int32(n_rows), Int32(n_cols),
            grid_dim=(n_cols, 1, 1), block_dim=(STATS_TPB, 1, 1),
        )
    ctx.enqueue_function[shift_columns_kernel](
        m.unsafe_ptr(), mu.unsafe_ptr(), Int32(n_rows), Int32(n_cols), Float32(-1.0),
        grid_dim=((cells + 255) // 256, 1, 1), block_dim=(256, 1, 1),
    )
    ctx.enqueue_function[square_in_place_kernel](
        m.unsafe_ptr(), Int32(cells),
        grid_dim=((cells + 255) // 256, 1, 1), block_dim=(256, 1, 1),
    )
    comptime if IDN_DECOMP_MEAN_LAUNCH:
        column_mean_launch(ctx, var_out, m, n_rows, n_cols)
    else:
        ctx.enqueue_function[column_mean_kernel](
            var_out.unsafe_ptr(), m.unsafe_ptr(), Int32(n_rows), Int32(n_cols),
            grid_dim=(n_cols, 1, 1), block_dim=(STATS_TPB, 1, 1),
        )


@always_inline
def _mp(buf: DeviceBuffer[DType.float32]) -> MutPointer[Float32, MutAnyOrigin]:
    return MutPointer[Float32, MutAnyOrigin](unsafe_from_address=Int(buf.unsafe_ptr()))


def tsvd_finish_kernel(
    var_t: MutPointer[Float32, MutAnyOrigin], nc: Int32, var_x: MutPointer[Float32, MutAnyOrigin], nf: Int32,
    explained: MutPointer[Float32, MutAnyOrigin], ratio: MutPointer[Float32, MutAnyOrigin],
):
    """`tsvd_finish_host` (decomposition/tsvd_finish.mojo) in one block of
    TSVD_FIN_TPB threads: thread t's slot, the same halving tree, then the
    divisions strided over the threads."""
    var t = Int(thread_idx.x)
    var sh = stack_allocation[TSVD_FIN_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var sl = stack_allocation[TSVD_FIN_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var s = tsvd_slot_sum(var_x, Int(nf), t)
    sh[t] = s.hi
    sl[t] = s.lo
    barrier()
    var half = TSVD_FIN_TPB // 2
    while half > 0:
        if t < half:
            var u = ff_add(FF(sh[t], sl[t]), FF(sh[t + half], sl[t + half]))
            sh[t] = u.hi
            sl[t] = u.lo
        barrier()
        half //= 2
    var full = ff_f32(FF(sh[0], sl[0]))
    var i = t
    while i < Int(nc):
        explained.unsafe_store(i, var_t.unsafe_load(i))
        ratio.unsafe_store(i, tsvd_ratio(var_t, i, full))
        i += TSVD_FIN_TPB


def tsvd_explained_host(
    ctx: DeviceContext,
    x_ptr: MutPointer[Float32, MutUntrackedOrigin],
    components_ptr: MutPointer[Float32, MutUntrackedOrigin],
    explained_ptr: MutPointer[Float32, MutUntrackedOrigin],
    ratio_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_rows: Int,
    n_features: Int,
    n_components: Int,
) raises:
    """scikit-learn TruncatedSVD's `explained_variance_` (np.var of X V^T
    per column, ddof 0) and `explained_variance_ratio_` (against the summed
    column variances of X), cuML `tsvdFitTransform`'s definition, in this
    binding (lane/algos-decomp, 2026-09-27; it had run through the x_decomp
    expansion binding). X V^T is `tsvd_transform_host`'s `gemm_nt`."""
    pca_validate(n_rows, n_features, n_components)
    var x = ctx.enqueue_create_buffer[DType.float32](n_rows * n_features)
    var components = ctx.enqueue_create_buffer[DType.float32](n_components * n_features)
    var xt = ctx.enqueue_create_buffer[DType.float32](n_rows * n_components)
    var mu_t = ctx.enqueue_create_buffer[DType.float32](n_components)
    var var_t = ctx.enqueue_create_buffer[DType.float32](n_components)
    var mu_x = ctx.enqueue_create_buffer[DType.float32](n_features)
    var var_x = ctx.enqueue_create_buffer[DType.float32](n_features)
    ctx.enqueue_copy(dst_buf=x, src_ptr=x_ptr)
    ctx.enqueue_copy(dst_buf=components, src_ptr=components_ptr)
    ctx.synchronize()
    gemm_nt(ctx, xt, x, components, n_rows, n_components, n_features)
    _column_variance(ctx, xt, mu_t, var_t, n_rows, n_components)
    _column_variance(ctx, x, mu_x, var_x, n_rows, n_features)
    # the tail on the device (decomposition/tsvd_finish.mojo, the host
    # column's statements; cpu-gpu-cleanup c-decomp)
    var d_exp = ctx.enqueue_create_buffer[DType.float32](max(n_components, 1))
    var d_rat = ctx.enqueue_create_buffer[DType.float32](max(n_components, 1))
    ctx.enqueue_function[tsvd_finish_kernel](
        _mp(var_t), Int32(n_components), _mp(var_x), Int32(n_features), _mp(d_exp), _mp(d_rat),
        grid_dim=1, block_dim=TSVD_FIN_TPB,
    )
    if n_components > 0:
        ctx.enqueue_copy(dst_ptr=explained_ptr, src_buf=d_exp)
        ctx.enqueue_copy(dst_ptr=ratio_ptr, src_buf=d_rat)
    ctx.synchronize()
    _ = d_exp^
    _ = d_rat^


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


# --------------------------------------------------------------------------
# THE WHITENED PAIR (DEVIATIONS 580-584; the arithmetic lives in
# `decomposition/impl/linalg/detail/pca.mojo`).
#
# TWO NEW SURFACES RATHER THAN TWO CHANGED ONES, DELIBERATELY. Whitening
# needs two things the unwhitened surfaces do not take -- the fit's singular
# values and the fit's row count -- and widening `pca_transform_host` /
# `inverse_transform_host` would change the arity of two functions that
# `bindings/_mojolearn_estimators.mojo` already exports. Adding beside them
# keeps every existing call site and every shipped binding byte for byte
# where it was, and makes the Python layer's capability test a plain
# `hasattr` on the new name.
# --------------------------------------------------------------------------


def pca_whiten_transform_host(
    ctx: DeviceContext,
    x_ptr: MutPointer[Float32, MutUntrackedOrigin],
    mean_ptr: MutPointer[Float32, MutUntrackedOrigin],
    components_ptr: MutPointer[Float32, MutUntrackedOrigin],
    singular_ptr: MutPointer[Float32, MutUntrackedOrigin],
    out_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_rows: Int,
    n_features: Int,
    n_components: Int,
    n_fit_rows: Int,
) raises:
    """`pcaTransform` with `prms.whiten = true`.

    `n_fit_rows` is the row count of the matrix the model was FITTED on, not
    of `x`. That is DEVIATION 580 and it is the one place this surface
    disagrees with cuML's dense path on purpose; the deviation note in
    `pca.mojo` carries the three-way evidence.
    """
    var x = ctx.enqueue_create_buffer[DType.float32](n_rows * n_features)
    var mu = ctx.enqueue_create_buffer[DType.float32](n_features)
    var components = ctx.enqueue_create_buffer[DType.float32](
        n_components * n_features
    )
    var components_w = ctx.enqueue_create_buffer[DType.float32](
        n_components * n_features
    )
    var singular = ctx.enqueue_create_buffer[DType.float32](n_components)
    var out = ctx.enqueue_create_buffer[DType.float32](n_rows * n_components)
    ctx.enqueue_copy(dst_buf=x, src_ptr=x_ptr)
    ctx.enqueue_copy(dst_buf=mu, src_ptr=mean_ptr)
    ctx.enqueue_copy(dst_buf=components, src_ptr=components_ptr)
    ctx.enqueue_copy(dst_buf=singular, src_ptr=singular_ptr)
    ctx.synchronize()
    var trace = IdentityTrace()
    whiten_components(
        ctx, components, components_w, singular,
        n_components, n_features, n_fit_rows, False,
    )
    if trace.enabled:
        trace.header(
            String("pca.whiten n=") + String(n_rows) + " d="
            + String(n_features) + " n_components=" + String(n_components)
            + " n_fit_rows=" + String(n_fit_rows)
        )
        trace.record_device(
            ctx, "pca.whiten.components", components_w,
            n_components * n_features,
        )
    pca_transform(
        ctx, x, mu, components_w, out, n_rows, n_features, n_components
    )
    if trace.enabled:
        trace.record_device(
            ctx, "pca.whiten.scores", out, n_rows * n_components
        )
    ctx.enqueue_copy(dst_ptr=out_ptr, src_buf=out)
    ctx.synchronize()


def pca_whiten_inverse_transform_host(
    ctx: DeviceContext,
    scores_ptr: MutPointer[Float32, MutUntrackedOrigin],
    components_ptr: MutPointer[Float32, MutUntrackedOrigin],
    singular_ptr: MutPointer[Float32, MutUntrackedOrigin],
    mean_ptr: MutPointer[Float32, MutUntrackedOrigin],
    out_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_rows: Int,
    n_features: Int,
    n_components: Int,
    n_fit_rows: Int,
) raises:
    """`pcaInverseTransform` with `prms.whiten = true`.

    The mean is ALWAYS added back here, because there is no whitened
    truncated SVD: `TruncatedSVD` does not center and does not take a
    `whiten` argument in cuML or in scikit-learn, so the `add_mean` flag the
    unwhitened surface carries has nothing to select between on this one.
    """
    var scores = ctx.enqueue_create_buffer[DType.float32](
        n_rows * n_components
    )
    var components = ctx.enqueue_create_buffer[DType.float32](
        n_components * n_features
    )
    var components_w = ctx.enqueue_create_buffer[DType.float32](
        n_components * n_features
    )
    var components_t = ctx.enqueue_create_buffer[DType.float32](
        n_features * n_components
    )
    var singular = ctx.enqueue_create_buffer[DType.float32](n_components)
    var mean = ctx.enqueue_create_buffer[DType.float32](n_features)
    var out = ctx.enqueue_create_buffer[DType.float32](n_rows * n_features)
    ctx.enqueue_copy(dst_buf=scores, src_ptr=scores_ptr)
    ctx.enqueue_copy(dst_buf=components, src_ptr=components_ptr)
    ctx.enqueue_copy(dst_buf=singular, src_ptr=singular_ptr)
    ctx.enqueue_copy(dst_buf=mean, src_ptr=mean_ptr)
    ctx.synchronize()
    whiten_components(
        ctx, components, components_w, singular,
        n_components, n_features, n_fit_rows, True,
    )
    ctx.enqueue_function[transpose_kernel](
        components_t.unsafe_ptr(), components_w.unsafe_ptr(),
        Int32(n_components), Int32(n_features),
        grid_dim=((n_features + TRANSPOSE_TILE - 1) // TRANSPOSE_TILE,
                  (n_components + TRANSPOSE_TILE - 1) // TRANSPOSE_TILE, 1),
        block_dim=(TRANSPOSE_TILE, TRANSPOSE_TILE, 1),
    )
    ctx.synchronize()
    gemm_nt(ctx, out, scores, components_t, n_rows, n_features, n_components)
    ctx.enqueue_function[shift_columns_kernel](
        out.unsafe_ptr(), mean.unsafe_ptr(), Int32(n_rows), Int32(n_features),
        Float32(1.0), grid_dim=((n_rows * n_features + 255) // 256, 1, 1),
        block_dim=(256, 1, 1),
    )
    ctx.enqueue_copy(dst_ptr=out_ptr, src_buf=out)
    ctx.synchronize()
