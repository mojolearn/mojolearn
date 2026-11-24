# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Typed GPU entries behind `bindings/_mojolearn_mixture.mojo`'s Python doors
(lane thin-trees-metrics, 2026-10-21: the `_gmm_*_run` bodies moved word for
word out of that shell; no Python here)."""
from std.memory import unsafe_memcpy
from core.hostptr import copy_f32
from mixture.estimator import (
    GaussianMixtureModel,
    GmmParams,
    gaussian_mixture_fit,
    gaussian_mixture_predict,
    gaussian_mixture_predict_proba,
    gaussian_mixture_sample,
    gaussian_mixture_score,
    gaussian_mixture_score_samples,
    gaussian_mixture_bic_of,
    gaussian_mixture_aic_of,
    gaussian_mixture_fit_device,
    gaussian_mixture_score_samples_device,
    gaussian_mixture_predict_proba_device,
    gaussian_mixture_predict_device,
    gaussian_mixture_score_device,
)
from mixture.multi_gpu import gmm_device_plan
from max.gpu.host import DeviceBuffer, DeviceContext


def _gmm_fit_run(
    x: List[Float32],
    n: Int,
    d: Int,
    params: GmmParams,
    wp: MutPointer[Float32, MutUntrackedOrigin],
    mp: MutPointer[Float32, MutUntrackedOrigin],
    cp: MutPointer[Float32, MutUntrackedOrigin],
    pp: MutPointer[Float32, MutUntrackedOrigin],
    lp: MutPointer[Float32, MutUntrackedOrigin],
    sp: MutPointer[Float64, MutUntrackedOrigin],
    labels_address: Int = 0,
) raises -> Int:
    # `labels_address` non-zero: fit_predict's labels (n int32 words) are
    # written by the fit itself (mixture/estimator.mojo, N401).
    var model = gaussian_mixture_fit(x, n, d, params, labels_address=labels_address)
    return _gmm_fit_words(model, d, wp, mp, cp, pp, lp, sp)


def _gmm_fit_device_run(
    ctx: DeviceContext,
    mut dx: DeviceBuffer[DType.float32],
    n: Int,
    d: Int,
    params: GmmParams,
    wp: MutPointer[Float32, MutUntrackedOrigin],
    mp: MutPointer[Float32, MutUntrackedOrigin],
    cp: MutPointer[Float32, MutUntrackedOrigin],
    pp: MutPointer[Float32, MutUntrackedOrigin],
    lp: MutPointer[Float32, MutUntrackedOrigin],
    sp: MutPointer[Float64, MutUntrackedOrigin],
    labels_address: Int = 0,
) raises -> Int:
    """`_gmm_fit_run` on an X already on the mixture context (a resident
    slot, lane vdf-scope-cluster2b): the same fit body, the same words out."""
    var model = gaussian_mixture_fit_device(
        gmm_device_plan(), ctx, dx, n, d, params, labels_address=labels_address
    )
    return _gmm_fit_words(model, d, wp, mp, cp, pp, lp, sp)


def _gmm_fit_words(
    model: GaussianMixtureModel,
    d: Int,
    wp: MutPointer[Float32, MutUntrackedOrigin],
    mp: MutPointer[Float32, MutUntrackedOrigin],
    cp: MutPointer[Float32, MutUntrackedOrigin],
    pp: MutPointer[Float32, MutUntrackedOrigin],
    lp: MutPointer[Float32, MutUntrackedOrigin],
    sp: MutPointer[Float64, MutUntrackedOrigin],
) raises -> Int:
    """The fitted model into the caller's buffers; returns n_iter."""
    var k = model.n_components
    copy_f32(model.weights.unsafe_ptr(), wp, k)
    copy_f32(model.means.unsafe_ptr(), mp, k * d)
    copy_f32(model.covariances.unsafe_ptr(), cp, k * d * d)
    copy_f32(model.precisions_cholesky.unsafe_ptr(), pp, k * d * d)
    copy_f32(model.log_det_chol.unsafe_ptr(), lp, k)
    sp.unsafe_store(0, Float64(model.n_iter))
    var conv = 0
    if model.converged:
        conv = 1
    sp.unsafe_store(1, Float64(conv))
    sp.unsafe_store(2, Float64(model.lower_bound))
    return model.n_iter


def _gmm_score_samples_run(
    model: GaussianMixtureModel,
    x: List[Float32],
    n: Int,
    op: MutPointer[Float32, MutUntrackedOrigin],
) raises:
    var s = gaussian_mixture_score_samples(model, x, n)
    copy_f32(s.unsafe_ptr(), op, n)


def _gmm_predict_proba_run(
    model: GaussianMixtureModel,
    x: List[Float32],
    n: Int,
    op: MutPointer[Float32, MutUntrackedOrigin],
) raises:
    var r = gaussian_mixture_predict_proba(model, x, n)
    copy_f32(r.unsafe_ptr(), op, n * model.n_components)


def _gmm_predict_run(
    model: GaussianMixtureModel,
    x: List[Float32],
    n: Int,
    op: MutPointer[Int32, MutUntrackedOrigin],
) raises:
    var labels = gaussian_mixture_predict(model, x, n)
    # One block copy into the caller's int32 buffer (was a per-row store
    # loop; p3 rank 98).
    if n > 0:
        unsafe_memcpy(dest=op, src=MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(labels.unsafe_ptr())), count=n)
    _ = labels^


def _gmm_sample_run(
    model: GaussianMixtureModel,
    n: Int,
    seed: UInt64,
    xp: MutPointer[Float32, MutUntrackedOrigin],
    yp: MutPointer[Int32, MutUntrackedOrigin],
) raises:
    var labels = List[Int32](length=max(0, n), fill=Int32(0))
    var x = gaussian_mixture_sample(model, n, seed, labels)
    copy_f32(x.unsafe_ptr(), xp, n * model.n_features)
    if n > 0:
        unsafe_memcpy(dest=yp, src=MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(labels.unsafe_ptr())), count=n)
    _ = labels^


def _gmm_score_bic_aic_run(
    model: GaussianMixtureModel,
    x: List[Float32],
    n: Int,
    sp: MutPointer[Float64, MutUntrackedOrigin],
) raises:
    # ONE scoring E-step for the three words (lane vdf-scope-cluster2b: bic
    # and aic re-ran the whole score, two more uploads and E-steps of X).
    # The same score word feeds both formulas: the same bits.
    var sc = gaussian_mixture_score(model, x, n)
    sp.unsafe_store(0, Float64(sc))
    sp.unsafe_store(1, Float64(gaussian_mixture_bic_of(model, sc, n)))
    sp.unsafe_store(2, Float64(gaussian_mixture_aic_of(model, sc, n)))


def _gmm_score_device_run(
    operation: Int,
    model: GaussianMixtureModel,
    ctx: DeviceContext,
    mut dx: DeviceBuffer[DType.float32],
    n: Int,
    destination: Int,
) raises:
    """The four scoring runs on an X already on the mixture context (a
    resident slot, lane vdf-scope-cluster2b): `operation` 0 score_samples,
    1 predict_proba, 2 predict, 3 score / bic / aic (the descriptor's
    MIXTURE_NATIVE_SCORE_SAMPLES .. SCORE_BIC_AIC less 3). The same bodies
    as the host runs above, the same words out."""
    if operation == 0:
        var s = gaussian_mixture_score_samples_device(gmm_device_plan(), model, ctx, dx, n)
        copy_f32(s.unsafe_ptr(), MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=destination), n)
    elif operation == 1:
        var r = gaussian_mixture_predict_proba_device(gmm_device_plan(), model, ctx, dx, n)
        copy_f32(r.unsafe_ptr(), MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=destination),
                 n * model.n_components)
    elif operation == 2:
        var labels = gaussian_mixture_predict_device(gmm_device_plan(), model, ctx, dx, n)
        if n > 0:
            unsafe_memcpy(dest=MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=destination),
                          src=MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(labels.unsafe_ptr())),
                          count=n)
        _ = labels^
    else:
        var sp = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=destination)
        var sc = gaussian_mixture_score_device(gmm_device_plan(), model, ctx, dx, n)
        sp.unsafe_store(0, Float64(sc))
        sp.unsafe_store(1, Float64(gaussian_mixture_bic_of(model, sc, n)))
        sp.unsafe_store(2, Float64(gaussian_mixture_aic_of(model, sc, n)))
