# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""PCA.transform over a model-selection fold resident on the device (lane
a6-device-x-family-classical, 2026-11-09).

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
from decomposition.impl.linalg.detail.pca import pca_transform


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
    n_components). `ctx` is `msel_device_x_context()`. Returns (finite word,
    range word): both 0 when every gathered word is finite and every index
    in range; otherwise the door refuses and discards `out_ptr`'s rows."""
    var x = ctx.enqueue_create_buffer[DType.float32](n_rows * n_features)
    var mu = ctx.enqueue_create_buffer[DType.float32](n_features)
    var components = ctx.enqueue_create_buffer[DType.float32](n_components * n_features)
    var out = ctx.enqueue_create_buffer[DType.float32](n_rows * n_components)
    var status = ctx.enqueue_create_buffer[DType.int32](2)
    enqueue_fill[DType.int32](ctx, status, Int32(0))
    msel_device_x_f32(ctx, base_id, base_rows, n_features, index_id, n_rows, x, status.unsafe_ptr(), 0)
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
