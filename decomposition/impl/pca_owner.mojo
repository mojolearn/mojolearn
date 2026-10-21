# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`PCA.fit` for the exact solvers as ONE native operation on the GPU binding
(lane activate-pca-owner, 2026-10-21): the fit, the MLE rank and the rank
selection that `decomposition.PCA.fit` used to sequence from Python as three
native calls (`pca_fit` / `pca_fit_full`, `x_decomp_dev_pca_mle_rank` and
`pca_rank_finish`) run here, in the same order, on the same kernels.

SAME BITS AS THE SEQUENCE IT REPLACES. The fit is `pca_fit_host` /
`pca_fit_full_host` unchanged. The MLE rank runs the six launches of
`x_decomp/select_dev.mojo::dev_pca_mle_rank_run` on this binding's context
with the same grid shapes; its input is the exact binary64 widening of the
float32 spectrum (`sf64_from_f32`, the words `array('d')` held), done on the
device instead of in Python. The rank and the dropped tail's noise variance
are `pca_rank_finish`, which the Python form called.

`pca_randomized_owned` is `svd_solver='randomized'`: the decomp lane's
randomized driver (x_decomp/pca_randomized.mojo) on its resident matrices.

With an output address, the same call projects X onto the kept components
(`fit_transform`): `pca_transform_host`, or `pca_whiten_transform_host` with
its device refusals, exactly what `transform` runs, on the fitted arrays.

The caller allocates the five outputs (and the projection) at the fitted
component count; the kept components are their prefix (C order), so the
Python layer only trims.
"""
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import bitcast
from max.gpu.host import DeviceContext

from checks.soft_f64 import sf64_from_f32
from decomposition.estimator import (
    pca_fit_full_host,
    pca_fit_host,
    pca_transform_host,
    pca_whiten_transform_host,
)
from decomposition.impl.linalg.detail.svd_full import pca_full_validate
from decomposition.impl.pca_rank import pca_rank_finish
from x_decomp.cells import F32Ptr
from x_decomp.device import DevExec
from x_decomp.pca_randomized import pca_randomized_fit_run
from x_decomp.pca_randomized_dev import DevRDev
from x_decomp.pca_mle import (
    MLE_P,
    mle_base_kernel,
    mle_cand_kernel,
    mle_ll_kernel,
    mle_part_kernel,
    mle_pick_kernel,
    mle_prep_kernel,
    mle_scratch_words,
)

#: how `n_components` picks the kept count after the fit
comptime PCA_KEEP_COUNT = 0  # an int (or None): every fitted component
comptime PCA_KEEP_FRACTION = 1  # a float in (0, 1): the explained-variance fraction
comptime PCA_KEEP_MLE = 2  # "mle": Minka's MLE of the rank

#: x_decomp/device.mojo TPB and `_blocks`: the MLE launches keep their grids
comptime _MLE_TPB = 128


def _mle_blocks(count: Int) -> Int:
    return (count + _MLE_TPB - 1) // _MLE_TPB if count > 0 else 1


def _f32_at(addr: Int) -> F32Ptr:
    return F32Ptr(unsafe_from_address=addr)


def pca_mle_widen_kernel(ev: F32Ptr, d: Int32, sp: F32Ptr):
    """sp[2k], sp[2k + 1] = the binary64 words (low first) of ev[k] widened
    exactly: the bytes `array('d', spectrum)` held."""
    var k = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if k < Int(d):
        var v = sf64_from_f32(ev.unsafe_load(k))
        sp.unsafe_store(2 * k, bitcast[DType.float32](UInt32(v & UInt64(0xFFFFFFFF))))
        sp.unsafe_store(2 * k + 1, bitcast[DType.float32](UInt32(v >> UInt64(32))))


def pca_mle_rank_device(
    ctx: DeviceContext, ev: MutPointer[Float32, MutUntrackedOrigin], d: Int, n: Int
) raises -> Int:
    """Minka's MLE rank of the d-value spectrum at host `ev` (n samples):
    `dev_pca_mle_rank_run`'s phases on `ctx`; 0 below two values."""
    if d < 2:
        return 0
    var spec = ctx.enqueue_create_buffer[DType.float32](d)
    var wide = ctx.enqueue_create_buffer[DType.float32](2 * d)
    var scratch = ctx.enqueue_create_buffer[DType.float32](mle_scratch_words(d))
    var rank = ctx.enqueue_create_buffer[DType.float32](1)
    ctx.enqueue_copy(dst_buf=spec, src_ptr=ev)
    var pe = _f32_at(Int(spec.unsafe_ptr()))
    var psp = _f32_at(Int(wide.unsafe_ptr()))
    var ps = _f32_at(Int(scratch.unsafe_ptr()))
    var pd = _f32_at(Int(rank.unsafe_ptr()))
    var d32 = Int32(d)
    var n32 = Int32(n)
    ctx.enqueue_function[pca_mle_widen_kernel](pe, d32, psp, grid_dim=_mle_blocks(d), block_dim=_MLE_TPB)
    ctx.enqueue_function[mle_prep_kernel](psp, d32, ps, grid_dim=_mle_blocks(d), block_dim=_MLE_TPB)
    ctx.enqueue_function[mle_base_kernel](psp, d32, n32, ps, grid_dim=_mle_blocks(d - 1), block_dim=_MLE_TPB)
    ctx.enqueue_function[mle_part_kernel](
        psp, d32, n32, ps, grid_dim=_mle_blocks((d - 1) * MLE_P), block_dim=_MLE_TPB
    )
    ctx.enqueue_function[mle_ll_kernel](psp, d32, n32, ps, grid_dim=_mle_blocks(d), block_dim=_MLE_TPB)
    ctx.enqueue_function[mle_cand_kernel](d32, ps, grid_dim=_mle_blocks(MLE_P), block_dim=_MLE_TPB)
    ctx.enqueue_function[mle_pick_kernel](d32, ps, pd, grid_dim=1, block_dim=1)
    var host = ctx.enqueue_create_host_buffer[DType.float32](1)
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=rank)
    ctx.synchronize()
    var picked = Int(host.unsafe_ptr().unsafe_load(0))
    _ = spec^
    _ = wide^
    _ = scratch^
    _ = rank^
    return picked


def pca_project_owned(
    ctx: DeviceContext,
    x_ptr: MutPointer[Float32, MutUntrackedOrigin],
    mean_ptr: MutPointer[Float32, MutUntrackedOrigin],
    components_ptr: MutPointer[Float32, MutUntrackedOrigin],
    singular_ptr: MutPointer[Float32, MutUntrackedOrigin],
    out_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_rows: Int,
    n_features: Int,
    n_components: Int,
    whiten: Bool,
) raises:
    """`transform(X)` of the model just fitted on X: the unwhitened
    projection, or the whitened one scaled by the fit's own row count with
    the whitened binding's dimension and device refusals. Nothing to write
    when no component is kept."""
    if whiten:
        # `_pca_whiten_apply`'s dimension refusals, in its words
        if n_rows < 1 or n_features < 1 or n_components < 1 or n_components > n_features or n_rows < 2:
            raise Error("PCA whitening requires positive rows/features/components and fit_rows>=2")
        if n_rows > 2147483647 or n_features > 2147483647 or n_components > 2147483647:
            raise Error("PCA whitening dimensions exceed Int32")
        if n_rows > 2147483647 // n_features or n_components > 2147483647 // n_features:
            raise Error("PCA whitening matrix cell count exceeds Int32")
        pca_whiten_transform_host(
            ctx, x_ptr, mean_ptr, components_ptr, singular_ptr, out_ptr,
            n_rows, n_features, n_components, n_rows, device_checks=True,
        )
    elif n_components > 0:
        pca_transform_host(ctx, x_ptr, mean_ptr, components_ptr, out_ptr, n_rows, n_features, n_components)
    ctx.synchronize()


def pca_fit_owned(
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
    dense: Bool,
    keep_mode: Int,
    frac: Float64,
    out_ptr: MutPointer[Float32, MutUntrackedOrigin],
    project: Bool,
    whiten: Bool,
) raises -> Tuple[Int, Float64]:
    """The whole exact fit: the five arrays written at `n_components`, then
    (for a float or "mle" `n_components`) the kept count and the dropped
    tail's noise variance, then with `project` X's projection onto the kept
    components. Returns (kept count, noise variance)."""
    if keep_mode != PCA_KEEP_COUNT and keep_mode != PCA_KEEP_FRACTION and keep_mode != PCA_KEEP_MLE:
        raise Error("pca_fit_owned: unknown n_components mode")
    var noise: Float64
    if dense:
        pca_full_validate(n_rows, n_features, n_components)
        noise = pca_fit_full_host(
            ctx, x_ptr, components_ptr, mean_ptr, explained_ptr, ratio_ptr, singular_ptr,
            n_rows, n_features, n_components,
        )
    else:
        noise = pca_fit_host(
            ctx, x_ptr, components_ptr, mean_ptr, explained_ptr, ratio_ptr, singular_ptr,
            n_rows, n_features, n_components,
        )
    ctx.synchronize()
    var keep = n_components
    if keep_mode != PCA_KEEP_COUNT:
        var keep_in = -1
        if keep_mode == PCA_KEEP_MLE:
            keep_in = pca_mle_rank_device(ctx, explained_ptr, n_components, n_rows)
        noise = 0.0
        keep = 0
        if n_components > 0:
            keep = pca_rank_finish(ratio_ptr, explained_ptr, n_components, keep_in, frac, noise)
    if project:
        pca_project_owned(
            ctx, x_ptr, mean_ptr, components_ptr, singular_ptr, out_ptr, n_rows, n_features, keep, whiten
        )
    return (keep, noise)


def pca_randomized_owned(
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
    oversamples: Int,
    iterations: Int,
    auto_iterations: Bool,
    seed: Int,
    out_ptr: MutPointer[Float32, MutUntrackedOrigin],
    project: Bool,
    whiten: Bool,
) raises -> Float64:
    """`svd_solver='randomized'`: the five arrays written at `n_components`
    by the decomp lane's resident driver, then with `project` X's
    projection. Returns the noise variance."""
    var noise = pca_randomized_fit_run[DevExec, DevRDev, True](
        _f32_at(Int(x_ptr)), _f32_at(Int(components_ptr)), _f32_at(Int(mean_ptr)),
        _f32_at(Int(explained_ptr)), _f32_at(Int(ratio_ptr)), _f32_at(Int(singular_ptr)),
        n_rows, n_features, n_components, oversamples, iterations, auto_iterations, seed,
    )
    if project:
        pca_project_owned(
            ctx, x_ptr, mean_ptr, components_ptr, singular_ptr, out_ptr, n_rows, n_features, n_components, whiten
        )
    return noise
