#!/usr/bin/env python3
"""Classical host inference gate (the classical host inference lane,
2026-09-13): LinearRegression, Ridge, TruncatedSVD, LogisticRegression and
PCA predicted on a CPU from a model fitted on a GPU, compared bit for bit;
since the kde svc host lane (2026-09-14) also KernelDensity, SVC and the
whitened PCA (`pca-whiten`, an identity_break lane of its own), and since the
knn host inference lane (2026-09-14) NearestNeighbors (`kneighbors`: distances
and indices), KNeighborsClassifier (`predict`, `predict_proba`) and
KNeighborsRegressor (`predict`), lanes knn, knn-clf and knn-reg, through
`mojolearn/host/_mojolearn_core_host.so`; since the neighbors and density
inference lane (2026-09-15) also the k-NN metric, ball cover and weighted
lanes, RadiusNeighbors (lanes radius, radius-manhattan, radius-chebyshev,
radius-minkowski-p3) and the KDE kernel, metric and weight lanes.

Two halves over tools/identity_break.py's own nine fixtures, so the
held-out rows here ARE the rows behind the `infer` column of the committed
GPU JSONs, and the digest this tool prints as `identity_hash` is that
column's cell (`identity_break._h` over the lane's probe outputs).

  record <dir> --lanes ols,ridge,...   on a GPU box, through the normal
           package: for every lane and every fixture, fit exactly what the
           identity_break lane fits (`LANES[lane]`), save the model as
           <dir>/<lane>/<fixture>/model.npz, predict the held-out rows on
           the GPU, reload the file through the class's own `load` and
           require the reload to predict the same bits, and write
           expected.json with the SHA-256, dtype and shape of every probe
           output, the identity_break hash, the vendor, GPU arch, numeric
           mode and model file hash. Refuses a CPU-only install, so the
           host binding can never record its own answer as the reference.
  check <dir>... [--gpu-column JSON ...]   on the CPU box, or on a GPU box
           through the host subclasses of `mojolearn._classical_host`
           (`_bind` answers the CPU binding, everything else is the GPU
           class's own Python): regenerate the held-out rows, verify them
           against the recorded hash, load model.npz through
           `mojolearn.host_model`, predict, and require every SHA-256,
           dtype and shape to equal the recording. With --gpu-column, the
           identity_break hash is also compared with that JSON's
           `cells[lane/fixture].infer` cell, one line per vendor, so one
           Mac run is judged against every committed GPU column.

Exit 0: every byte equal. 1: any mismatch. 2: the gate could not run.
--expect-mismatch inverts the verdict for the sabotage build
(MOJOLEARN_HOST_DIR pointing at a set built with
-D MOJOLEARN_HOST_SABOTAGE=1 and MOJOLEARN_HOST_ALLOW_SABOTAGE=1): exit 0
only if something differs.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))

from forest_host_gate import (  # noqa: E402
    digest_prediction, git_commit, host_info, sha256_bytes,
)

#: Held-out rows each lane probes, `identity_break.py`'s `Xh[:256]`.
PROBE_ROWS = 256


def _kernel_variant_probe(model, X, method):
    """Use the kernel-variant harness's exact held-out input contract."""
    import numpy as np
    held = np.ascontiguousarray(X[:64, :4] * np.float32(0.125))
    return (getattr(model, method)(held),)


def _forecast_pair(e):
    """identity_break's forecaster infer probe, the same call and the same
    byte check (`_same_bytes`), so its hash is that column's cell."""
    ib = identity_tool()
    h = ib.FORECAST_HORIZON
    return ib._same_bytes("forecast(h)", e.forecast(h),
                          "predict(n_obs, n_obs + h)", e.predict(e.n_obs_, e.n_obs_ + h))


def _radius_probe(e, X):
    """identity_break's radius infer probe: `_ragged` over the sorted query
    of the first 64 held-out rows."""
    return identity_tool()._ragged(e.radius_neighbors(X[:64], sort_results=True))


def _approximate_predict(e, X):
    """identity_break's hdbscan infer probe: approximate_predict's labels and
    probabilities on the held-out rows' first four columns, then
    membership_vector on the same rows and all_points_membership_vectors
    (the probe since HDBSCAN's soft clustering, 2026-09-15)."""
    from mojolearn.hdbscan import (
        all_points_membership_vectors, approximate_predict, membership_vector,
    )
    return tuple(approximate_predict(e, X[:, :4])) + (
        membership_vector(e, X[:, :4]), all_points_membership_vectors(e))


#: The k-means saved-model lanes (lane/classical-host-recordings, 2026-09-16).
#: One format, `mojolearn-kmeans-1`, carries every metric and every start, so
#: all six FITTED k-means lanes load through it. The cosine lane was NOT here
#: while it existed (deleted 2026-09-18, lane/kmeans-cosine-capability):
#: its fit is refused by name (cluster/impl/kmeans_params.mojo::validate), and
#: a refusal has no model to save, so it has no saved-model cell to record.
_KMEANS_LANES = ('kmeans', 'kmeans-random', 'kmeans-array', 'kmeans-weighted',
                 'kmeans-sqrt', 'kmeans-classic-pp')


def _kmeans_probe(e, X, kind):
    """identity_break's `_km_probe` infer pair, `(predict(Xh), transform(Xh))`,
    over the WHOLE held-out draw (LANE_PROBE_ROWS), so its hash is that
    column's `infer` cell.

    `_km_probe` also asserts, before returning, that `predict` over the
    TRAINING rows is `labels_` bit for bit. That assertion is about the fit,
    and a model reloaded from a file has no training rows; the saved-model
    restatement of it is the `predict_training_rows` extra below, which
    rebuilds those rows from the fixture and is compared against the `labels`
    extra, the array the file carries.
    """
    return (e.predict(X), e.transform(X))


#: The surfaces a loaded k-means model answers beside its identity cell. The
#: training-row predict is the claim that matters for a saved model: a fit's
#: own final assignment, reproduced from the file on a machine with no GPU.
_KMEANS_EXTRAS = {
    'transform': lambda e, X, kind: e.transform(X),
    'labels': lambda e, X, kind: e.labels_,
    'predict_training_rows': lambda e, X, kind: e.predict(identity_tool().fixture(kind)[0]),
}


#: Lanes whose probe needs the fixture KIND beside the held-out rows
#: (lane/saved-model-reference-gaps, 2026-09-16). Every other probe is called
#: `probe(model, Xh)`; these are called `probe(model, Xh, kind)`.
KIND_PROBES = ('spectral-precomputed',) + tuple(_KMEANS_LANES)


def _spectral_precomputed_probe(e, X, kind):
    """identity_break's `spectral-precomputed` infer probe: predict on the
    affinity of the held-out rows to that lane's 1000 training rows, under
    `_cross_affinity`'s rule with the training matrix's threshold.

    identity_break stashes the matrix on the FITTED estimator
    (`_identity_heldout_affinity`); a model reloaded from disk carries no fit
    rows at all for a precomputed affinity, so this rebuilds it from the
    fixture instead of reading the attribute. `do_record` requires the hash it
    produces to equal identity_break's own, which is what would catch a
    rebuild that did not reproduce those bytes.
    """
    ib = identity_tool()
    P = ib.fixture(kind)[0][:1000, :4]
    return (e.predict(ib._cross_affinity(X[:, :4], P)),)


#: The surfaces every ARIMA lane adds beside its identity probe.
_ARIMA_EXTRAS = {
    'predict_in_sample': lambda e, X: e.predict(0, e.n_obs_),
    'predict_straddle': lambda e, X: e.predict(e.n_obs_ - 16, e.n_obs_ + 16),
    'params': lambda e, X: e.params_,
    'sigma2': lambda e, X: e.sigma2_,
}


def _arima_exog_future(X, rows):
    """identity_break's arima-exog regressors over the first `rows` held-out
    rows (`_arima_exog_block`), the future values the probe hands in."""
    return identity_tool()._arima_exog_block(X, rows)


#: The arima-exog lanes' surfaces (lane/arima-exog, 2026-09-15): the ARIMA
#: extras, with the straddle's 16 future rows of regressors from the held-out
#: draw, and the regression coefficients a loaded model answers.
_ARIMA_EXOG_EXTRAS = dict(
    _ARIMA_EXTRAS,
    predict_straddle=lambda e, X: e.predict(e.n_obs_ - 16, e.n_obs_ + 16, exog=_arima_exog_future(X, 16)),
    beta=lambda e, X: e.beta_,
)


def _hw_forecast_pair(e):
    """identity_break's Holt-Winters infer probe: forecast(H) through both
    return paths, the flat buffer and the `index=0` strided read, held to the
    same bytes (`_same_bytes`), so its hash is that column's cell."""
    ib = identity_tool()
    h = ib.FORECAST_HORIZON
    return ib._same_bytes("forecast(h)", e.forecast(h), "forecast(h, index=0)", e.forecast(h, index=0))


#: The surfaces every Holt-Winters lane adds beside its identity probe
#: (lane/inference-holtwinters, 2026-09-15): the in-sample one-step
#: predictions, a prediction straddling the end of the series, and the fitted
#: state a loaded model answers.
_HW_EXTRAS = {
    'predict_in_sample': lambda e, X: e.predict(0, e.n),
    'predict_straddle': lambda e, X: e.predict(e.n - 16, e.n + 16),
    'level': lambda e, X: e.level_,
    'trend': lambda e, X: e.trend_,
    'season': lambda e, X: e.season_,
    'alpha': lambda e, X: e.alpha_,
    'beta': lambda e, X: e.beta_,
    'gamma': lambda e, X: e.gamma_,
    'sse': lambda e, X: e.sse_,
}


def _ivf_probe(e, Q):
    """identity_break's ivf infer probe: `search` then `n_candidates_`."""
    d, i = e.search(Q)
    return (d, i, e.n_candidates_)


_IVF_EXTRAS = {
    'search_second_batch_ids': lambda e, X: e.search(X[64:128])[1],
    'list_indices': lambda e, X: e.list_indices_,
    'list_data': lambda e, X: e.list_data_,
}


def _embedding_probe(e, X):
    """identity_break's embedding infer probe: the 512 held-out ids (the
    first 512 bytes of the held-out rows, which `Xh[:256]` holds)."""
    import numpy as np
    ib = identity_tool()
    V, T = e.num_embeddings, 512
    ids = (ib._ids(X, 1, T).reshape(T) % V).astype(np.int32)
    return (np.asarray(e.forward(ids)),)


#: lane -> (estimator, identity_break probe, extra surfaces). The identity
#: probe is the tuple `identity_break` hashes for the `infer` column, in its
#: order (the knn lanes probe `Xh[:64]`, as `identity_break` does); the
#: first element is digested under PROBE_NAMES, the extras are hashed on
#: their own and are not part of that cell.
LANES = {
    'ols': ('LinearRegression', lambda e, X: (e.predict(X),), {}),
    'ridge': ('Ridge', lambda e, X: (e.predict(X),), {}),
    'tsvd': ('TruncatedSVD', lambda e, X: (e.transform(X),), {}),
    'logistic': ('LogisticRegression', lambda e, X: (e.predict_proba(X),),
                 {'predict': lambda e, X: e.predict(X),
                  'decision_function': lambda e, X: e.decision_function(X)}),
    'pca': ('PCA', lambda e, X: (e.transform(X),), {}),
    # The kde svc host lane (2026-09-14). kde's identity_break lane fits and
    # probes the first four columns; svc's probe is the pair (decision,
    # predict), the labels hashed as an extra too; pca-whiten adds the
    # whitened inverse as an extra, the second half of the host pair.
    'kde': ('KernelDensity', lambda e, X: (e.score_samples(X[:, :4]),), {}),
    'svc': ('SVC', lambda e, X: (e.decision_function(X), e.predict(X)),
            {'predict': lambda e, X: e.predict(X)}),
    'pca-whiten': ('PCA', lambda e, X: (e.transform(X),),
                   {'inverse_transform': lambda e, X: e.inverse_transform(e.transform(X))}),
    'knn': ('NearestNeighbors', lambda e, X: e.kneighbors(X[:64]),
            {'kneighbors_indices': lambda e, X: e.kneighbors(X[:64])[1]}),
    'knn-clf': ('KNeighborsClassifier',
                lambda e, X: (e.predict(X[:64]), e.predict_proba(X[:64])),
                {'predict_proba': lambda e, X: e.predict_proba(X[:64])}),
    'knn-reg': ('KNeighborsRegressor', lambda e, X: (e.predict(X[:64]),), {}),
    # lane/logistic-multiclass (2026-09-14): three classes through the
    # softmax loss; the probe is the pair (predict_proba, predict), the
    # order of the identity_break lane body in
    # docs/lanes/BRIEF_logistic_multiclass_2026-09-14.md section 5.
    'logistic-multiclass': ('LogisticRegression',
                            lambda e, X: (e.predict_proba(X), e.predict(X)),
                            {'predict': lambda e, X: e.predict(X),
                             'decision_function': lambda e, X: e.decision_function(X)}),
    # The neighbors and density inference lane (2026-09-15): every k-NN
    # metric and the ball cover arm, the distance-weighted vote and mean, the
    # radius query on its four metrics and the KDE kernel and metric pairs,
    # each probed exactly as its identity_break lane probes the held-out
    # rows (the radius probe is identity_break's `_ragged` over the sorted
    # query; the cosine KDE pair shifts its rows by 8, as the lane does).
    'knn-sqeuclidean': ('NearestNeighbors', lambda e, X: e.kneighbors(X[:64]),
                               {'kneighbors_indices': lambda e, X: e.kneighbors(X[:64])[1]}),
    'knn-manhattan': ('NearestNeighbors', lambda e, X: e.kneighbors(X[:64]),
                             {'kneighbors_indices': lambda e, X: e.kneighbors(X[:64])[1]}),
    'knn-chebyshev': ('NearestNeighbors', lambda e, X: e.kneighbors(X[:64]),
                             {'kneighbors_indices': lambda e, X: e.kneighbors(X[:64])[1]}),
    'knn-cosine': ('NearestNeighbors', lambda e, X: e.kneighbors(X[:64]),
                          {'kneighbors_indices': lambda e, X: e.kneighbors(X[:64])[1]}),
    'knn-minkowski-p3': ('NearestNeighbors', lambda e, X: e.kneighbors(X[:64]),
                                {'kneighbors_indices': lambda e, X: e.kneighbors(X[:64])[1]}),
    'knn-rbc': ('NearestNeighbors', lambda e, X: e.kneighbors(X[:64]),
                       {'kneighbors_indices': lambda e, X: e.kneighbors(X[:64])[1]}),
    'knn-clf-distance': ('KNeighborsClassifier',
                         lambda e, X: (e.predict(X[:64]), e.predict_proba(X[:64])),
                         {'predict_proba': lambda e, X: e.predict_proba(X[:64])}),
    'knn-reg-distance': ('KNeighborsRegressor', lambda e, X: (e.predict(X[:64]),), {}),
    'radius': ('RadiusNeighbors', lambda e, X: _radius_probe(e, X), {}),
    'radius-manhattan': ('RadiusNeighbors', lambda e, X: _radius_probe(e, X), {}),
    'radius-chebyshev': ('RadiusNeighbors', lambda e, X: _radius_probe(e, X), {}),
    'radius-minkowski-p3': ('RadiusNeighbors', lambda e, X: _radius_probe(e, X), {}),
    'kde-tophat-sqeuclidean': ('KernelDensity', lambda e, X: (e.score_samples(X[:, :4]),), {}),
    'kde-epanechnikov-l1': ('KernelDensity', lambda e, X: (e.score_samples(X[:, :4]),), {}),
    'kde-exponential-chebyshev': ('KernelDensity', lambda e, X: (e.score_samples(X[:, :4]),), {}),
    'kde-linear-cosine': ('KernelDensity', lambda e, X: (e.score_samples(X[:, :4] + 8.0),), {}),
    'kde-cosine-minkowski': ('KernelDensity', lambda e, X: (e.score_samples(X[:, :4]),), {}),
    'kde-weighted': ('KernelDensity', lambda e, X: (e.score_samples(X[:, :4]),), {}),
    # IsolationForest (same lane): identity_break scores every held-out row
    # and predicts the first 512, so these two lanes probe the whole
    # held-out draw (LANE_PROBE_ROWS) rather than its first PROBE_ROWS.
    'iforest': ('IsolationForest', lambda e, X: (e.score_samples(X), e.predict(X[:512])),
               {'predict': lambda e, X: e.predict(X[:512]),
                'decision_function': lambda e, X: e.decision_function(X)}),
    'iforest-tuned': ('IsolationForest', lambda e, X: (e.score_samples(X), e.predict(X[:512])),
                     {'predict': lambda e, X: e.predict(X[:512]),
                      'decision_function': lambda e, X: e.decision_function(X)}),
    # GaussianMixture (same lane), through the inference-only mixture
    # binding: 64 held-out rows of the first four columns, as the lanes ask.
    'gmm': ('GaussianMixture',
            lambda e, X: (e.score_samples(X[:64, :4]), e.predict(X[:64, :4]), e.predict_proba(X[:64, :4])),
            {'predict': lambda e, X: e.predict(X[:64, :4]),
             'predict_proba': lambda e, X: e.predict_proba(X[:64, :4])}),
    'gmm-random-init': ('GaussianMixture', lambda e, X: (e.score_samples(X[:64, :4]),),
                        {'predict': lambda e, X: e.predict(X[:64, :4])}),
    # GaussianMixture.sample (same lane, after main's gmm-sample lanes): the
    # identity_break probe is sample(1024)'s (X, y) from the saved model; it
    # reads no held-out rows.
    'gmm-sample': ('GaussianMixture', lambda e, X: tuple(e.sample(1024)),
                   {'sample_y': lambda e, X: e.sample(1024)[1]}),
    'gmm-random-init-sample': ('GaussianMixture', lambda e, X: tuple(e.sample(1024)),
                               {'sample_y': lambda e, X: e.sample(1024)[1]}),
    # GaussianProcessRegressor (same lane), through the inference-only gp
    # binding: the predictive mean and std of 64 held-out rows of four
    # columns, as every GP lane asks, normalize_y's scale-back included.
    'gp': ('GaussianProcessRegressor', lambda e, X: tuple(e.predict(X[:64, :4], return_std=True)),
          {'std': lambda e, X: e.predict(X[:64, :4], return_std=True)[1]}),
    'gp-matern12': ('GaussianProcessRegressor', lambda e, X: tuple(e.predict(X[:64, :4], return_std=True)),
                   {'std': lambda e, X: e.predict(X[:64, :4], return_std=True)[1]}),
    'gp-matern32': ('GaussianProcessRegressor', lambda e, X: tuple(e.predict(X[:64, :4], return_std=True)),
                   {'std': lambda e, X: e.predict(X[:64, :4], return_std=True)[1]}),
    'gp-matern52-ard': ('GaussianProcessRegressor', lambda e, X: tuple(e.predict(X[:64, :4], return_std=True)),
                       {'std': lambda e, X: e.predict(X[:64, :4], return_std=True)[1]}),
    'gp-normalize-y': ('GaussianProcessRegressor', lambda e, X: tuple(e.predict(X[:64, :4], return_std=True)),
                      {'std': lambda e, X: e.predict(X[:64, :4], return_std=True)[1]}),
    # GaussianProcessClassifier (same lane, after lane/gaussian-process-
    # classifier merged): predict and predict_proba of 64 held-out rows of
    # four columns, the pair both gpc lanes hash, through gpc_predict.
    'gpc': ('GaussianProcessClassifier', lambda e, X: (e.predict(X[:64, :4]), e.predict_proba(X[:64, :4])),
           {'predict_proba': lambda e, X: e.predict_proba(X[:64, :4])}),
    'gpc-multiclass': ('GaussianProcessClassifier', lambda e, X: (e.predict(X[:64, :4]), e.predict_proba(X[:64, :4])),
                      {'predict_proba': lambda e, X: e.predict_proba(X[:64, :4])}),
    # HDBSCAN (same lane): approximate_predict's labels and probabilities,
    # membership_vector on the first 256 held-out rows of four columns and
    # all_points_membership_vectors, the tuple both lanes hash.
    'hdbscan': ('HDBSCAN', lambda e, X: _approximate_predict(e, X),
               {'probabilities': lambda e, X: _approximate_predict(e, X)[1],
                'membership_vector': lambda e, X: _approximate_predict(e, X)[2],
                'all_points_membership_vectors': lambda e, X: _approximate_predict(e, X)[3]}),
    'hdbscan-leaf': ('HDBSCAN', lambda e, X: _approximate_predict(e, X),
                    {'probabilities': lambda e, X: _approximate_predict(e, X)[1],
                     'membership_vector': lambda e, X: _approximate_predict(e, X)[2],
                     'all_points_membership_vectors': lambda e, X: _approximate_predict(e, X)[3]}),
    # lane/inference-forecast-umap-pca (2026-09-15). pca-full-whiten is the
    # dense SVD fit with the whitened pair. umap probes identity_break's
    # batch of 64 held-out rows in one call: the transform's answer depends
    # on the batch, so the claim is the same bytes for the same batch.
    'pca-full-whiten': ('PCA', lambda e, X: (e.transform(X),),
                        {'inverse_transform': lambda e, X: e.inverse_transform(e.transform(X))}),
    'umap': ('UMAP', lambda e, X: (e.transform(X[:64, :8]),), {}),
    # The forecasters take no rows: the probe is identity_break's pair,
    # forecast(H) and predict(n_obs, n_obs + H), held to the same bytes. The
    # extras are what the CPU inference surface adds beyond that cell: the
    # in-sample prediction, a prediction straddling the end of the series,
    # and the fitted-state accessors of the loaded model.
    'arima': ('ARIMA', lambda e, X: _forecast_pair(e), dict(
        _ARIMA_EXTRAS, ar=lambda e, X: e.ar_, mu=lambda e, X: e.mu_)),
    'arima-011': ('ARIMA', lambda e, X: _forecast_pair(e), dict(
        _ARIMA_EXTRAS, ma=lambda e, X: e.ma_)),
    'arima-seasonal-c': ('ARIMA', lambda e, X: _forecast_pair(e), dict(
        _ARIMA_EXTRAS, ar=lambda e, X: e.ar_, sar=lambda e, X: e.sar_, mu=lambda e, X: e.mu_)),
    # lane/arima-exog (2026-09-15): the probe is identity_break's
    # `_arima_exog_probe` over the held-out draw's first FORECAST_HORIZON
    # rows (LANE_PROBE_ROWS), so its hash is that column's cell.
    'arima-exog': ('ARIMA', lambda e, X: identity_tool()._arima_exog_probe(e, X), dict(
        _ARIMA_EXOG_EXTRAS, ar=lambda e, X: e.ar_, mu=lambda e, X: e.mu_)),
    'arima-exog-seasonal': ('ARIMA', lambda e, X: identity_tool()._arima_exog_probe(e, X), dict(
        _ARIMA_EXOG_EXTRAS, ar=lambda e, X: e.ar_, sar=lambda e, X: e.sar_)),
    # lane/inference-holtwinters (2026-09-15): the saved Holt-Winters models,
    # additive and multiplicative, through the forecast inference binding.
    'holtwinters': ('ExponentialSmoothing', lambda e, X: _hw_forecast_pair(e), _HW_EXTRAS),
    'holtwinters-multiplicative': ('ExponentialSmoothing', lambda e, X: _hw_forecast_pair(e), _HW_EXTRAS),
    # lane/inference-linear-svm (2026-09-15): the option variants of ols,
    # ridge and logistic through the formats above, and the scalers,
    # coordinate descent and the kernel methods through formats of their
    # own. Each probe is its identity_break lane's infer probe; the extras
    # are the other public surfaces of the loaded model.
    'ols-no-intercept': ('LinearRegression', lambda e, X: (e.predict(X),), {}),
    'ols-weighted': ('LinearRegression', lambda e, X: (e.predict(X),), {}),
    'ridge-no-intercept': ('Ridge', lambda e, X: (e.predict(X),), {}),
    'logistic-l1': ('LogisticRegression', lambda e, X: (e.predict_proba(X),),
                    {'predict': lambda e, X: e.predict(X),
                     'decision_function': lambda e, X: e.decision_function(X)}),
    'logistic-elasticnet': ('LogisticRegression', lambda e, X: (e.predict_proba(X),),
                            {'predict': lambda e, X: e.predict(X),
                             'decision_function': lambda e, X: e.decision_function(X)}),
    'logistic-unpenalized-no-intercept': ('LogisticRegression', lambda e, X: (e.predict_proba(X),),
                                          {'predict': lambda e, X: e.predict(X),
                                           'decision_function': lambda e, X: e.decision_function(X)}),
    'standard-scaler': ('StandardScaler', lambda e, X: (e.transform(X),),
                        {'inverse_transform': lambda e, X: e.inverse_transform(e.transform(X))}),
    'standard-scaler-no-mean': ('StandardScaler', lambda e, X: (e.transform(X),),
                                {'inverse_transform': lambda e, X: e.inverse_transform(e.transform(X))}),
    'standard-scaler-no-std': ('StandardScaler', lambda e, X: (e.transform(X),),
                               {'inverse_transform': lambda e, X: e.inverse_transform(e.transform(X))}),
    'minmax-scaler': ('MinMaxScaler', lambda e, X: (e.transform(X),),
                      {'inverse_transform': lambda e, X: e.inverse_transform(e.transform(X))}),
    'minmax-scaler-clip': ('MinMaxScaler', lambda e, X: (e.transform(X),),
                           {'inverse_transform': lambda e, X: e.inverse_transform(e.transform(X))}),
    'lasso': ('Lasso', lambda e, X: (e.predict(X),), {}),
    'elasticnet': ('ElasticNet', lambda e, X: (e.predict(X),), {}),
    'elasticnet-l2end-no-intercept': ('ElasticNet', lambda e, X: (e.predict(X),), {}),
    'kernel-ridge': ('KernelRidge', lambda e, X: (e.predict(X[:64, :4]),), {}),
    'nystroem': ('Nystroem', lambda e, X: (e.transform(X[:64, :4]),), {}),
    **{f'kernel-ridge-{kernel}': (
        'KernelRidge', lambda e, X: _kernel_variant_probe(e, X, 'predict'), {})
       for kernel in ('poly', 'sigmoid', 'laplacian')},
    **{f'nystroem-{kernel}': (
        'Nystroem', lambda e, X: _kernel_variant_probe(e, X, 'transform'), {})
       for kernel in ('poly', 'sigmoid', 'laplacian')},
    'rbf-sampler': ('RBFSampler', lambda e, X: (e.transform(X),), {}),
    # lane/inference-embedding-ivf-cholesky (2026-09-15). The IVF lanes probe
    # identity_break's 64 held-out queries over the saved, GPU-built index
    # (distances, ids, candidate counts); the extras search a second query
    # batch and read the index arrays back. The embedding lane's probe is
    # identity_break's 512 held-out ids through the saved table.
    'ivf': ('IVFIndex', lambda e, X: _ivf_probe(e, X[:64]), _IVF_EXTRAS),
    'ivf-euclidean': ('IVFIndex', lambda e, X: _ivf_probe(e, X[:64]), _IVF_EXTRAS),
    'embedding': ('Embedding', lambda e, X: _embedding_probe(e, X), {}),
    # Stage 2 of the same lane: an index built and EXTENDED on the GPU, saved;
    # the extras extend a clone of the loaded index by 64 held-out rows (the
    # host binding's extend on a CPU) and search 64 more rows after it.
    'ivf-extend': ('IVFIndex', lambda e, X: _ivf_probe(e, X[:64]), dict(
        _IVF_EXTRAS,
        extend_lists=lambda e, X: e._clone().extend(X[64:128]).extend_labels_,
        extend_list_indices=lambda e, X: e._clone().extend(X[64:128]).list_indices_,
        search_after_extend_ids=lambda e, X: e._clone().extend(X[64:128]).search(X[128:192])[1],
    )),
    # lane/inference-svm (2026-09-15): SVC's linear and polynomial kernels
    # through the svc format, and SVR (rbf and linear) through its own. Each
    # probe is its identity_break lane's infer probe.
    'svc-linear': ('SVC', lambda e, X: (e.decision_function(X), e.predict(X)),
                   {'predict': lambda e, X: e.predict(X)}),
    'svc-poly': ('SVC', lambda e, X: (e.decision_function(X), e.predict(X)),
                 {'predict': lambda e, X: e.predict(X)}),
    'svr': ('SVR', lambda e, X: (e.predict(X),), {}),
    'svr-linear': ('SVR', lambda e, X: (e.predict(X),), {}),
    # lane/saved-model-reference-gaps (2026-09-16): the saved-model route of
    # the three predicts that shipped on 2026-09-15 and that no gate covered.
    # DBSCAN and AgglomerativeClustering are DEVIATION 2740
    # (lane/inference-transductive-predict), SpectralClustering is DEVIATION
    # 2860 (lane/spectral-predict). Each probe is its identity_break lane's
    # infer probe, on the first four columns of the held-out rows.
    'dbscan': ('DBSCAN', lambda e, X: (e.predict(X[:, :4]),), {}),
    'agglomerative': ('AgglomerativeClustering', lambda e, X: (e.predict(X[:, :4]),), {}),
    'spectral': ('SpectralClustering', lambda e, X: (e.predict(X[:, :4]),), {}),
    # The precomputed arm predicts on an (n_new, n_train) affinity, not on
    # rows, so its probe is built from the fixture as well as the held-out
    # draw; see KIND_PROBES.
    'spectral-precomputed': ('SpectralClustering', _spectral_precomputed_probe, {}),
    # lane/classical-host-recordings (2026-09-16): the k-means saved-model
    # route. lane/kmeans-save gave KMeans `save` and `load` and put
    # `mojolearn-kmeans-1` in `_classical_host._FORMATS`; until this table
    # carried the lanes there could be no recording for them, exactly as
    # there could be none for the four predict lanes before 2026-09-16.
    'kmeans': ('KMeans', _kmeans_probe, _KMEANS_EXTRAS),
    'kmeans-random': ('KMeans', _kmeans_probe, _KMEANS_EXTRAS),
    'kmeans-array': ('KMeans', _kmeans_probe, _KMEANS_EXTRAS),
    'kmeans-weighted': ('KMeans', _kmeans_probe, _KMEANS_EXTRAS),
    'kmeans-sqrt': ('KMeans', _kmeans_probe, _KMEANS_EXTRAS),
    'kmeans-classic-pp': ('KMeans', _kmeans_probe, _KMEANS_EXTRAS),
}
PROBE_NAMES = {'ols': 'predict', 'ridge': 'predict', 'tsvd': 'transform',
               'logistic': 'predict_proba', 'pca': 'transform',
               'kde': 'score_samples', 'svc': 'decision_function',
               'pca-whiten': 'transform',
               'knn': 'kneighbors_distances', 'knn-clf': 'predict',
               'knn-reg': 'predict', 'logistic-multiclass': 'predict_proba',
               'pca-full-whiten': 'transform', 'umap': 'transform',
               'arima': 'forecast', 'arima-011': 'forecast', 'arima-seasonal-c': 'forecast',
               'arima-exog': 'forecast', 'arima-exog-seasonal': 'forecast',
               'holtwinters': 'forecast', 'holtwinters-multiplicative': 'forecast',
               'ols-no-intercept': 'predict', 'ols-weighted': 'predict',
               'ridge-no-intercept': 'predict', 'logistic-l1': 'predict_proba',
               'logistic-elasticnet': 'predict_proba',
               'logistic-unpenalized-no-intercept': 'predict_proba',
               'standard-scaler': 'transform', 'standard-scaler-no-mean': 'transform',
               'standard-scaler-no-std': 'transform', 'minmax-scaler': 'transform',
               'minmax-scaler-clip': 'transform', 'lasso': 'predict', 'elasticnet': 'predict',
               'elasticnet-l2end-no-intercept': 'predict', 'kernel-ridge': 'predict',
               'nystroem': 'transform', 'rbf-sampler': 'transform',
               **{f'kernel-ridge-{kernel}': 'predict' for kernel in ('poly', 'sigmoid', 'laplacian')},
               **{f'nystroem-{kernel}': 'transform' for kernel in ('poly', 'sigmoid', 'laplacian')},
               'ivf': 'search_distances', 'ivf-euclidean': 'search_distances', 'embedding': 'forward', 'ivf-extend': 'search_distances',
               'svc-linear': 'decision_function', 'svc-poly': 'decision_function',
               'svr': 'predict', 'svr-linear': 'predict',
               'dbscan': 'predict', 'agglomerative': 'predict',
               'spectral': 'predict', 'spectral-precomputed': 'predict',
               **{lane: 'predict' for lane in _KMEANS_LANES}}
PROBE_NAMES.update({lane: {'NearestNeighbors': 'kneighbors_distances', 'KNeighborsClassifier': 'predict',
                           'KNeighborsRegressor': 'predict', 'RadiusNeighbors': 'radius_neighbors_counts',
                           'KernelDensity': 'score_samples', 'IsolationForest': 'score_samples',
                           'GaussianMixture': 'score_samples', 'HDBSCAN': 'approximate_predict_labels',
                           'GaussianProcessRegressor': 'predict_mean',
                           'GaussianProcessClassifier': 'predict'}[spec[0]]
                    for lane, spec in LANES.items() if lane not in PROBE_NAMES})


def _fit_logistic_multiclass(ml, X, yc, yr, Xh=None):
    """The `logistic-multiclass` lane body, word for word the one handed to
    tools/identity_break.py's owner (the brief, section 5), carried here
    until that file has it; `do_record` uses it when `ib.LANES` lacks the
    lane. Three classes from the fixture's own labels: the binary rule plus
    one for rows whose column 5 is above its median (a column no fixture
    perturbs; the median split keeps all three classes on every fixture)."""
    import numpy as np
    ib = identity_tool()
    y3 = (yc + (X[:, 5] > np.median(X[:, 5]))).astype(np.int32)
    m = ml.LogisticRegression(max_iter=50).fit(X, y3)
    return ib._fit(dict(coef=ib._h(m.coef_), proba=ib._h(m.predict_proba(X[:256]))), m,
                   lambda e: (e.predict_proba(Xh[:256]), e.predict(Xh[:256])))


#: lane -> fit, for a lane the gate knows before identity_break does.
LOCAL_FITS = {'logistic-multiclass': _fit_logistic_multiclass}


def identity_tool():
    import identity_break
    return identity_break


def package_root(args):
    if args.package_root is None:
        sys.path.insert(0, str(ROOT / 'python'))
    elif args.package_root:
        sys.path.insert(0, os.path.abspath(args.package_root))


#: Lanes whose identity_break probe reads the whole held-out draw; every
#: other lane reads its first PROBE_ROWS rows.
LANE_PROBE_ROWS = {'iforest': None, 'iforest-tuned': None,
                   # the regressors' future values over FORECAST_HORIZON rows
                   'arima-exog': 512, 'arima-exog-seasonal': 512,
                   # `_km_probe` predicts and transforms the whole held-out
                   # draw, not its first 256 rows; a 256-row probe here would
                   # hash something that is not the lane's `infer` cell and
                   # do_record would refuse it.
                   **{lane: None for lane in _KMEANS_LANES}}


def probe_rows(ib, lane, kind):
    """The held-out row count `lane` probes on fixture `kind`."""
    rows = LANE_PROBE_ROWS.get(lane, PROBE_ROWS)
    return int(ib.heldout(kind).shape[0]) if rows is None else rows


def held_out(ib, kind, lane=None):
    """The identity_break held-out slice the lane probes, as a numpy
    array, and its bytes' SHA-256."""
    Xh = ib.heldout(kind)[:probe_rows(ib, lane, kind)]
    return Xh, sha256_bytes(Xh.tobytes())


def digests_for(lane, model, Xh, ib, kind):
    """Every surface's `(sha256, dtype, shape)`, the identity_break hash of
    the identity probe, and the seconds spent."""
    _, probe, extras = LANES[lane]
    out = {}
    started = time.perf_counter()
    outputs = probe(model, Xh, kind) if lane in KIND_PROBES else probe(model, Xh)
    out['identity_hash'] = ib._h(*outputs)
    out[PROBE_NAMES[lane]] = dict(zip(('sha256', 'dtype', 'shape'), digest_prediction(outputs[0])))
    for name, fn in extras.items():
        extra = fn(model, Xh, kind) if lane in KIND_PROBES else fn(model, Xh)
        out[name] = dict(zip(('sha256', 'dtype', 'shape'), digest_prediction(extra)))
    out['seconds'] = round(time.perf_counter() - started, 6)
    return out


def do_record(args):
    package_root(args)
    ib = identity_tool()
    import mojolearn
    vendor = mojolearn.vendor()
    if vendor == 'cpu':
        print('gate: record must run on a GPU box; this install is CPU-only', file=sys.stderr)
        return 2
    if mojolearn.numeric_mode() != 'identical':
        print(f'gate: record requires MOJOLEARN_NUMERIC_MODE=identical, this process is '
              f'{mojolearn.numeric_mode()}', file=sys.stderr)
        return 2
    lanes = [l for l in args.lanes.split(',') if l]
    unknown = [l for l in lanes if l not in LANES]
    if unknown:
        print(f'gate: unknown lanes {unknown}; this gate knows {sorted(LANES)}', file=sys.stderr)
        return 2
    fixtures = [f for f in ib.FIXTURES if not args.fixtures or f in args.fixtures.split(',')]
    for lane in lanes:
        for kind in fixtures:
            directory = args.fixture_dir / lane / kind
            if directory.exists() and any(directory.iterdir()) and not args.overwrite:
                print(f'gate: {directory} exists and is not empty; pass --overwrite', file=sys.stderr)
                return 2
    for lane in lanes:
        estimator = LANES[lane][0]
        for kind in fixtures:
            directory = args.fixture_dir / lane / kind
            directory.mkdir(parents=True, exist_ok=True)
            X, yc, yr = ib.fixture(kind)
            Xh_full = ib.heldout(kind)
            fit = (ib.LANES[lane] if lane in ib.LANES else LOCAL_FITS[lane])(mojolearn, X, yc, yr, Xh_full)
            model = fit.est
            if type(model).__name__ != estimator:
                print(f'gate: lane {lane} fitted {type(model).__name__}, not {estimator}', file=sys.stderr)
                return 2
            Xh, x_sha = held_out(ib, kind, lane)
            gpu = digests_for(lane, model, Xh, ib, kind)
            # The identity_break probe on the fitted model must agree with
            # the tool's own infer cell for this fit, or the probe here is
            # not that column's.
            tool_hash = ib._h(*fit.probe(model))
            if tool_hash != gpu['identity_hash']:
                print(f'gate: {lane}/{kind} probe hash {gpu["identity_hash"]} is not the '
                      f'identity_break probe hash {tool_hash}', file=sys.stderr)
                return 2
            model_path = directory / 'model.npz'
            model.save(str(model_path))
            back = type(model).load(str(model_path))
            reload = digests_for(lane, back, Xh, ib, kind)
            for key in gpu:
                if key == 'seconds':
                    continue
                if gpu[key] != reload[key]:
                    print(f'gate: {lane}/{kind} {key} differs between the fitted model and its '
                          f'reload on the GPU path: {gpu[key]} vs {reload[key]}', file=sys.stderr)
                    return 1
            spec = dict(lane=lane, kind=kind, estimator=estimator, heldout_seed=ib.HELDOUT_SEED,
                        probe_rows=probe_rows(ib, lane, kind), x_sha256=x_sha)
            (directory / 'fixture.json').write_text(json.dumps(spec, indent=2, sort_keys=True) + '\n')
            report = dict(
                status='RECORDED', lane=lane, kind=kind, estimator=estimator, vendor=vendor,
                gpu_arch=mojolearn.gpu_arch(), numeric_mode=mojolearn.numeric_mode(),
                model_sha256=sha256_bytes(model_path.read_bytes()), x_sha256=x_sha,
                host=host_info(), commit=git_commit(),
                recorded_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), predictions=gpu,
                reload_equal=True,
            )
            (directory / 'expected.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
            print(f"record {lane} {kind} {estimator} identity_hash {gpu['identity_hash']} "
                  f"{PROBE_NAMES[lane]} {gpu[PROBE_NAMES[lane]]['sha256']} (vendor {vendor})")
    return 0


def gpu_columns(paths):
    cols = []
    for p in paths or []:
        with open(p) as fh:
            j = json.load(fh)
        cols.append((os.path.basename(p), j))
    return cols


def sabotage_verdict(verdict_ok, moved, unmoved, every_lane=False, every_fixture=False, lane_rule_only=()):
    """The --expect-mismatch verdict over `<lane>/<fixture>` names.

    Plain: one differing fixture anywhere is a catch. --every-lane: every lane
    must differ on at least one fixture. --every-fixture (lane/ties-sabotage,
    2026-09-15): every fixture of every lane must differ, except that a lane
    named in `lane_rule_only` keeps the --every-lane rule, by name. Returns
    (verdict, exit code, lines to print)."""
    # verdict_ok also includes optional GPU-column comparisons. A stale or
    # disagreeing column alone cannot demonstrate a changed CPU prediction.
    # Only differences from this saved model's own recorded outputs populate
    # moved; require that concrete evidence even under the plain any-fixture
    # rule. An empty run is never a successful negative control either.
    caught = bool(moved) and not verdict_ok
    verdict = 'EXPECTED MISMATCH SEEN' if caught else 'SABOTAGE NOT CAUGHT'
    code = 0 if caught else 1
    lines = []
    if not (every_lane or every_fixture):
        return verdict, code, lines
    exempt = set(lane_rule_only)
    moved_lanes = {m.split('/')[0] for m in moved}
    dead = sorted({u.split('/')[0] for u in unmoved} - moved_lanes)
    for name in unmoved:
        lane = name.split('/')[0]
        rule = 'every-lane by name' if every_fixture and lane in exempt else ('every-fixture' if every_fixture else 'every-lane')
        lines.append(f'check {name} did not move (its lane {"moved elsewhere" if lane in moved_lanes else "DID NOT MOVE"}; rule {rule})')
    if every_fixture:
        still = sorted(u for u in unmoved if u.split('/')[0] not in exempt)
        if still:
            return f'SABOTAGE NOT CAUGHT ON FIXTURES {",".join(still)}', 1, lines
    if dead:
        return f'SABOTAGE NOT CAUGHT ON LANES {",".join(dead)}', 1, lines
    return verdict, code, lines


def do_check(args):
    package_root(args)
    try:
        ib = identity_tool()
        import mojolearn
        from mojolearn._classical_host import binary_path, binary_paths, host_model
    except Exception as exc:
        print(f'gate: import failed: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 2
    columns = gpu_columns(args.gpu_column)
    dirs = []
    for root in args.fixture_dir:
        if (root / 'expected.json').exists():
            dirs.append(root)
        else:
            dirs.extend(sorted(p.parent for p in root.glob('*/*/expected.json')))
    if not dirs:
        print('gate: no fixture directory with an expected.json under the paths given', file=sys.stderr)
        return 2
    results = []
    verdict_ok = True
    moved, unmoved = [], []
    for directory in dirs:
        expected = json.loads((directory / 'expected.json').read_text())
        if expected.get('status') != 'RECORDED' or 'predictions' not in expected:
            print(f'gate: {directory} status is {expected.get("status")!r}; the GPU-side recording '
                  'is OWED (tools/classical_host_gate.py record on a GPU box)', file=sys.stderr)
            return 2
        spec = json.loads((directory / 'fixture.json').read_text())
        lane, kind = spec['lane'], spec['kind']
        if lane not in LANES or kind not in ib.FIXTURES or int(spec.get('probe_rows', 0)) != probe_rows(ib, lane, kind):
            print(f'gate: {directory} fixture.json names a lane, fixture or probe size this gate '
                  'does not know', file=sys.stderr)
            return 2
        Xh, x_sha = held_out(ib, kind, lane)
        if x_sha != spec.get('x_sha256') or x_sha != expected.get('x_sha256'):
            print(f'gate: {directory} regenerated held-out rows hash {x_sha}, the fixture records '
                  f'{spec.get("x_sha256")}', file=sys.stderr)
            return 2
        model_path = directory / 'model.npz'
        model_sha = sha256_bytes(model_path.read_bytes())
        if expected.get('model_sha256') != model_sha:
            print(f'gate: {model_path} hashes {model_sha}, expected.json records '
                  f'{expected.get("model_sha256")}', file=sys.stderr)
            return 2
        try:
            model = host_model(str(model_path))
            got = digests_for(lane, model, Xh, ib, kind)
        except Exception as exc:
            print(f'gate: {directory} host predict failed: {type(exc).__name__}: {exc}', file=sys.stderr)
            return 2
        if model.estimator != expected.get('estimator'):
            print(f'gate: {directory} model is {model.estimator}, expected.json says '
                  f'{expected.get("estimator")}', file=sys.stderr)
            return 2
        want = expected['predictions']
        cases = []
        for key in sorted(k for k in want if k not in ('seconds', 'identity_hash')):
            if key not in got:
                print(f'gate: {directory} {key} recorded but not computed here', file=sys.stderr)
                return 2
            equal = (want[key]['sha256'] == got[key]['sha256'] and want[key]['dtype'] == got[key]['dtype']
                     and list(want[key]['shape']) == list(got[key]['shape']))
            verdict_ok = verdict_ok and equal
            cases.append(dict(case=key, want=want[key]['sha256'], got=got[key]['sha256'],
                              want_dtype=want[key]['dtype'], got_dtype=got[key]['dtype'],
                              want_shape=want[key]['shape'], got_shape=got[key]['shape'], equal=equal))
            print(f"check {lane} {kind} {key} {'EQUAL' if equal else 'DIFFER'} "
                  f"gpu {want[key]['sha256']} host {got[key]['sha256']}")
        ih_equal = want['identity_hash'] == got['identity_hash']
        verdict_ok = verdict_ok and ih_equal
        print(f"check {lane} {kind} identity_hash {'EQUAL' if ih_equal else 'DIFFER'} "
              f"gpu {want['identity_hash']} host {got['identity_hash']}")
        if not ih_equal or not all(c['equal'] for c in cases):
            moved.append(f'{lane}/{kind}')
        else:
            unmoved.append(f'{lane}/{kind}')
        vendors = []
        for label, j in columns:
            cell = j.get('cells', {}).get(f'{lane}/{kind}')
            infer = (cell or {}).get('infer') or []
            theirs = infer[0] if infer else None
            equal = theirs == got['identity_hash']
            if theirs is None:
                status = 'ABSENT'
            elif isinstance(theirs, str) and theirs.startswith('n/a:'):
                # A record older than the lane's infer probe carries its
                # reason (`n/a:transductive` on hdbscan before 2026-09-15),
                # not a hash: nothing to compare, so it cannot differ.
                status = 'N/A'
                equal = None
            else:
                status = 'EQUAL' if equal else 'DIFFER'
                verdict_ok = verdict_ok and equal
            vendors.append(dict(column=label, vendor=j.get('vendor'), infer=theirs, equal=equal if theirs else None))
            print(f"column {lane} {kind} {label} {status} theirs {theirs} host {got['identity_hash']}")
        results.append(dict(lane=lane, kind=kind, estimator=model.estimator,
                            recorded_vendor=expected.get('vendor'), recorded_gpu_arch=expected.get('gpu_arch'),
                            recorded_numeric_mode=expected.get('numeric_mode'), model_sha256=model_sha,
                            identity_hash=dict(want=want['identity_hash'], got=got['identity_hash'], equal=ih_equal),
                            columns=vendors, seconds=got['seconds'], cases=cases))
    if args.expect_mismatch:
        verdict, code, lines = sabotage_verdict(
            verdict_ok, moved, unmoved, every_lane=args.every_lane,
            every_fixture=args.every_fixture, lane_rule_only=args.lane_rule_only)
        for line in lines:
            print(line)
    else:
        verdict = 'IDENTICAL' if verdict_ok else 'MISMATCH'
        code = 0 if verdict_ok else 1
    report = dict(verdict=verdict, expect_mismatch=bool(args.expect_mismatch), exit=code, unmoved=unmoved,
                  binary=binary_path(), binaries=binary_paths(), vendor=mojolearn.vendor(), host=host_info(),
                  gpu_columns=[label for label, _ in columns], commit=git_commit(),
                  checked_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), fixtures=results)
    if args.report:
        if args.report.exists():
            print(f'gate: {args.report} exists; refusing to overwrite a report', file=sys.stderr)
            return 2
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(f'gate verdict {verdict} ({len(results)} fixtures, {len(columns)} GPU columns, exit {code})')
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--package-root', default=None,
                        help="directory to import mojolearn from (default: this checkout's python/; "
                             "'' for the installed package)")
    sub = parser.add_subparsers(dest='command', required=True)
    rec = sub.add_parser('record', help='on a GPU box, fit, save and record every lane and fixture under a directory')
    rec.add_argument('fixture_dir', type=Path)
    rec.add_argument('--lanes', required=True, help='comma separated: ' + ','.join(LANES))
    rec.add_argument('--fixtures', default='', help='comma separated identity_break fixtures (default all nine)')
    rec.add_argument('--overwrite', action='store_true')
    chk = sub.add_parser('check', help='on the CPU box, compare the host predictions with every expected.json')
    # nargs='*' so an empty list reaches the refusal in main(), which says
    # what is missing, instead of argparse's usage line.
    chk.add_argument('fixture_dir', type=Path, nargs='*',
                     help='a <lane>/<fixture> directory, or a root holding <lane>/<fixture>/ directories')
    chk.add_argument('--gpu-column', action='append', default=[],
                     help='an identity_break JSON whose infer cells are compared too (repeatable)')
    chk.add_argument('--report', type=Path, help='new exclusive JSON report')
    chk.add_argument('--expect-mismatch', action='store_true')
    chk.add_argument('--every-lane', action='store_true',
                     help='with --expect-mismatch, every lane (not only one) must differ on at least one fixture')
    chk.add_argument('--every-fixture', action='store_true',
                     help='with --expect-mismatch, every fixture of every lane must differ')
    chk.add_argument('--lane-rule-only', action='append', default=[], metavar='LANE',
                     help='with --every-fixture, this lane keeps the --every-lane rule, by name (repeatable)')
    args = parser.parse_args()
    # `--lane-rule-only` belongs to the CHECK subparser only, so a `record`
    # Namespace has no such attribute and reading it unguarded raised
    # AttributeError before do_record ran a line. That is how the four predict
    # lanes came to have no recording: the tool that makes one had been
    # unusable since the flag was added (lane/ties-sabotage, 2026-09-15), and
    # the crash is in main(), ahead of every other refusal, so it reached a
    # rented GPU box on 2026-09-16 and cost it its recording phase
    # (lane/saved-model-reference-gaps).
    if getattr(args, 'lane_rule_only', None) and not args.every_fixture:
        parser.error('--lane-rule-only needs --every-fixture')
    if args.command == 'check' and not args.fixture_dir:
        print('classical_host_gate.py check: no fixture directory given, so there is nothing to check; '
              'in the CPU identity gate this is CLASSICAL_RECORDED or SAVED_MODEL_RECORDED, which the '
              'CPU surface manifest step writes and which is empty when that step did not run',
              file=sys.stderr)
        return 2
    if args.command == 'record':
        return do_record(args)
    return do_check(args)


if __name__ == '__main__':
    sys.exit(main())
