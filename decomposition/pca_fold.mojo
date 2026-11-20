# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""PCA.transform (lane a6-device-x-family-classical) and PCA's exact fit
(lane a8d-device-x-decomp-manifold) over a model-selection fold resident on
the device (2026-11-09).

`pca_transform_host` (decomposition/estimator.mojo) uploads X from host
words. Under cross-validation the fold's rows are already resident in model
selection's store (core/msel_device_x.mojo): `pca_transform_fold` gathers
them on the device straight into the transform's X buffer (no host gather,
no upload), with the finite and range tests in the same pass, uploads only
the fitted mean and components, runs the same `pca_transform` and returns
the two status words with the projected rows. Only GPU bindings import this
file (the store lives on the device).

Numerics: the same rows in the same order, the same kernels on the same
words as the host-address path: no bit changes.
"""
from max.gpu.host import DeviceContext
from core.device_zero import enqueue_fill
from core.msel_device_x import msel_device_x_f32
from decomposition.impl.linalg.detail.pca import pca_transform, pca_validate
from decomposition.impl.linalg.detail.svd_full import pca_full_validate
from decomposition.estimator import pca_fit_cov_device_x, pca_fit_full_device_x, tsvd_fit_device, _tsvd_explained_tail
from decomposition.impl.pca_owner import pca_keep_finish


struct PCAFoldFit(Copyable, Movable):
    """The fold fit's kept count, noise variance and its two status words
    (finite, range)."""
    var keep: Int
    var noise: Float64
    var finite: Int32
    var in_range: Int32

    def __init__(out self, keep: Int, noise: Float64, finite: Int32, in_range: Int32):
        self.keep = keep
        self.noise = noise
        self.finite = finite
        self.in_range = in_range


def pca_fit_fold(
    ctx: DeviceContext,
    base_id: Int,
    base_rows: Int,
    index_id: Int,
    components_ptr: MutPointer[Float32, MutUntrackedOrigin],
    mean_ptr: MutPointer[Float32, MutUntrackedOrigin],
    explained_ptr: MutPointer[Float32, MutUntrackedOrigin],
    ratio_ptr: MutPointer[Float32, MutUntrackedOrigin],
    singular_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_rows: Int,
    n_features: Int,
    n_components: Int,
    dense: Bool,
    keep_mode: Int,
    frac: Float64,
) raises -> PCAFoldFit:
    """PCA's exact fit (`pca_fit_owned` without a projection; lane
    a8d-device-x-decomp-manifold) of fold rows `index_id` of the resident
    base `base_id` (`base_rows` x `n_features`): the rows are gathered on the
    device straight into the fit's X buffer with the finite and range tests
    in the same pass (no host gather, no upload), the status words are read
    at the gather's synchronisation (the fit synchronises there anyway), and
    a flagged fold is not fit (the door refuses). Otherwise the same body as
    the host-address entry (`pca_fit_cov_device_x` / `pca_fit_full_device_x`,
    then `pca_keep_finish`): the same rows in the same order, no bit change.
    `ctx` is `msel_fold_entry_context()`."""
    if dense:
        pca_full_validate(n_rows, n_features, n_components)
    var x = ctx.enqueue_create_buffer[DType.float32](n_rows * n_features)
    var status = ctx.enqueue_create_buffer[DType.int32](2)
    enqueue_fill[DType.int32](ctx, status, Int32(0))
    msel_device_x_f32(ctx, base_id, base_rows, n_features, index_id, n_rows, x, status.unsafe_ptr(), 0)
    var flags = ctx.enqueue_create_host_buffer[DType.int32](2)
    ctx.enqueue_copy(dst_ptr=flags.unsafe_ptr(), src_buf=status)
    ctx.synchronize()
    var finite = flags.unsafe_ptr()[0]
    var in_range = flags.unsafe_ptr()[1]
    _ = flags^
    _ = status^
    if finite != 0 or in_range != 0:
        _ = x^
        return PCAFoldFit(0, 0.0, finite, in_range)
    var noise: Float64
    if dense:
        noise = pca_fit_full_device_x(
            ctx, x, components_ptr, mean_ptr, explained_ptr, ratio_ptr, singular_ptr,
            n_rows, n_features, n_components,
        )
    else:
        noise = pca_fit_cov_device_x(
            ctx, x, components_ptr, mean_ptr, explained_ptr, ratio_ptr, singular_ptr,
            n_rows, n_features, n_components, False,
        )
    ctx.synchronize()
    _ = x^
    var keep = pca_keep_finish(ctx, explained_ptr, ratio_ptr, n_rows, n_components, keep_mode, frac, noise)
    return PCAFoldFit(keep, noise, finite, in_range)


def tsvd_fit_fold(
    ctx: DeviceContext,
    base_id: Int,
    base_rows: Int,
    index_id: Int,
    components_ptr: MutPointer[Float32, MutUntrackedOrigin],
    singular_ptr: MutPointer[Float32, MutUntrackedOrigin],
    explained_ptr: MutPointer[Float32, MutUntrackedOrigin],
    ratio_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_rows: Int,
    n_features: Int,
    n_components: Int,
) raises -> Tuple[Int32, Int32]:
    """TruncatedSVD's Gram fit and its explained variance
    (`tsvd_fit_explained_host` under one upload; lane a8d) of fold rows
    `index_id` of the resident base: gathered on the device into the fit's
    X buffer with the finite and range tests in the same pass, the words
    read at that synchronisation, then `tsvd_fit_device(resident=True)` and
    the explained tail on the same buffer. Returns (finite word, range
    word); a flagged fold is not fit. The same rows in the same order, the
    same kernels: no bit change. `ctx` is `msel_fold_entry_context()`."""
    var x = ctx.enqueue_create_buffer[DType.float32](n_rows * n_features)
    var status = ctx.enqueue_create_buffer[DType.int32](2)
    enqueue_fill[DType.int32](ctx, status, Int32(0))
    msel_device_x_f32(ctx, base_id, base_rows, n_features, index_id, n_rows, x, status.unsafe_ptr(), 0)
    var flags = ctx.enqueue_create_host_buffer[DType.int32](2)
    ctx.enqueue_copy(dst_ptr=flags.unsafe_ptr(), src_buf=status)
    ctx.synchronize()
    var finite = flags.unsafe_ptr()[0]
    var in_range = flags.unsafe_ptr()[1]
    _ = flags^
    _ = status^
    if finite != 0 or in_range != 0:
        _ = x^
        return (finite, in_range)
    _ = tsvd_fit_device(
        ctx, x, components_ptr, components_ptr, singular_ptr, n_rows, n_features, n_components, 0, True
    )
    pca_validate(n_rows, n_features, n_components)
    _tsvd_explained_tail(ctx, x, components_ptr, explained_ptr, ratio_ptr, n_rows, n_features, n_components)
    _ = x^
    return (finite, in_range)


def pca_transform_fold(
    ctx: DeviceContext,
    base_id: Int,
    base_rows: Int,
    index_id: Int,
    mean_ptr: MutPointer[Float32, MutUntrackedOrigin],
    components_ptr: MutPointer[Float32, MutUntrackedOrigin],
    out_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_rows: Int,
    n_features: Int,
    n_components: Int,
) raises -> Tuple[Int32, Int32]:
    """The projection of fold rows `index_id` of the resident base
    `base_id` (`base_rows` x `n_features`) into `out_ptr` (n_rows x
    n_components). `ctx` is `msel_fold_entry_context()`. Returns (finite word,
    range word): both 0 when every gathered word is finite and every index
    in range; otherwise the door refuses and discards `out_ptr`'s rows."""
    var x = ctx.enqueue_create_buffer[DType.float32](n_rows * n_features)
    var mu = ctx.enqueue_create_buffer[DType.float32](n_features)
    var components = ctx.enqueue_create_buffer[DType.float32](n_components * n_features)
    var out = ctx.enqueue_create_buffer[DType.float32](n_rows * n_components)
    var status = ctx.enqueue_create_buffer[DType.int32](2)
    enqueue_fill[DType.int32](ctx, status, Int32(0))
    msel_device_x_f32(ctx, base_id, base_rows, n_features, index_id, n_rows, x, status.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin](), 0)
    ctx.enqueue_copy(dst_buf=mu, src_ptr=mean_ptr)
    ctx.enqueue_copy(dst_buf=components, src_ptr=components_ptr)
    pca_transform(ctx, x, mu, components, out, n_rows, n_features, n_components)
    # One synchronisation: the status words come back with the result
    # (core/admitted_finite_status.mojo contract). On a flagged fold the
    # door's fresh output is discarded by its refusal.
    var flags = ctx.enqueue_create_host_buffer[DType.int32](2)
    ctx.enqueue_copy(dst_ptr=flags.unsafe_ptr(), src_buf=status)
    ctx.enqueue_copy(dst_ptr=out_ptr, src_buf=out)
    ctx.synchronize()
    var finite = flags.unsafe_ptr()[0]
    var in_range = flags.unsafe_ptr()[1]
    _ = flags^
    _ = x^
    _ = mu^
    _ = components^
    _ = out^
    _ = status^
    return (finite, in_range)
