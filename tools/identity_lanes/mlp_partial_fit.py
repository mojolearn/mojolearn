# SPDX-License-Identifier: Apache-2.0
# Two optimizer updates made in separate public calls must match the uninterrupted fit.


def _mlp_partial_result(incremental, full, prediction, full_prediction):
    weights = np.concatenate([value.ravel() for value in (*incremental.coefs_, *incremental.intercepts_)])
    expected = np.concatenate([value.ravel() for value in (*full.coefs_, *full.intercepts_)])
    parts = dict(weights=_h(weights), prediction=_h(prediction), losses=_h(np.asarray(incremental.loss_curve_)),
                 progress=_h(np.asarray([incremental.n_iter_, incremental.t_], dtype=np.int64)),
                 training=incremental.training_digest_)
    mismatch = _mismatch_bytes("incremental MLP weights", weights, "uninterrupted MLP weights", expected)
    if mismatch:
        raise NumericalMismatch(mismatch, parts)
    if prediction.shape != full_prediction.shape or prediction.dtype != full_prediction.dtype:
        raise NumericalMismatch("incremental and uninterrupted MLP prediction shapes or dtypes differ", parts)
    mismatch = _mismatch_bytes("incremental MLP prediction", prediction, "uninterrupted MLP prediction", full_prediction)
    if mismatch:
        raise NumericalMismatch(mismatch, parts)
    if incremental.training_digest_ != full.training_digest_ or incremental.digest_ != full.digest_:
        raise NumericalMismatch("incremental MLP state differs from uninterrupted training", parts)
    return _fit(parts)


@lane("sequence-mlp-classifier-partial")
def _(ml, X, yc, yr, Xh=None):
    x = np.array([[1., .5], [-1., .25], [.75, -.5]], dtype=np.float32)
    y = np.array([4, 8, 9], dtype=np.int64)
    options = dict(hidden_layer_sizes=(2,), activation="tanh", solver="adam", max_iter=2,
                   batch_size=2, random_state=7, shuffle=True)
    incremental = ml.MLPClassifier(**options)
    incremental.partial_fit(x, y, classes=y)
    incremental.partial_fit(x, y)
    full = ml.MLPClassifier(**options).fit(x, y)
    return _mlp_partial_result(incremental, full, incremental.predict_proba(x), full.predict_proba(x))


@lane("sequence-mlp-regressor-partial")
def _(ml, X, yc, yr, Xh=None):
    x = np.array([[1., .5], [-1., .25], [.75, -.5]], dtype=np.float32)
    y = np.array([.25, .75, -1.], dtype=np.float32)
    options = dict(hidden_layer_sizes=(2,), activation="tanh", solver="sgd", learning_rate="invscaling",
                   learning_rate_init=.1, max_iter=2, batch_size=2, random_state=7, shuffle=True)
    incremental = ml.MLPRegressor(**options)
    incremental.partial_fit(x, y)
    incremental.partial_fit(x, y)
    full = ml.MLPRegressor(**options).fit(x, y)
    return _mlp_partial_result(incremental, full, incremental.predict(x), full.predict(x))
