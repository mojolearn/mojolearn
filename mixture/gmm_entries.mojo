# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Typed GPU entries behind `bindings/_mojolearn_mixture.mojo`'s Python doors
(lane thin-trees-metrics, 2026-10-21: the `_gmm_*_run` bodies moved word for
word out of that shell; no Python here)."""
from core.hostptr import copy_f32
from mixture.estimator import (
    GaussianMixtureModel,
    GmmParams,
    gaussian_mixture_aic,
    gaussian_mixture_bic,
    gaussian_mixture_fit,
    gaussian_mixture_predict,
    gaussian_mixture_predict_proba,
    gaussian_mixture_sample,
    gaussian_mixture_score,
    gaussian_mixture_score_samples,
)


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
) raises -> Int:
    var model = gaussian_mixture_fit(x, n, d, params)
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
    for i in range(n):
        op.unsafe_store(i, labels[i])


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
    for i in range(n):
        yp.unsafe_store(i, labels[i])


def _gmm_score_bic_aic_run(
    model: GaussianMixtureModel,
    x: List[Float32],
    n: Int,
    sp: MutPointer[Float64, MutUntrackedOrigin],
) raises:
    sp.unsafe_store(0, Float64(gaussian_mixture_score(model, x, n)))
    sp.unsafe_store(1, Float64(gaussian_mixture_bic(model, x, n)))
    sp.unsafe_store(2, Float64(gaussian_mixture_aic(model, x, n)))
