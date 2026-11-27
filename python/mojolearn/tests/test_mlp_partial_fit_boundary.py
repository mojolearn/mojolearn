import pickle

import numpy as np
import pytest

from mojolearn import MLPClassifier, MLPRegressor


def test_classifier_can_start_without_observing_all_declared_classes():
    x = np.array([[0, 1], [1, -1]], np.float32)
    estimator = MLPClassifier(hidden_layer_sizes=(2,), batch_size=1, random_state=13, device="cpu")
    assert estimator.partial_fit(x, ["cat", "cat"], classes=["owl", "cat", "dog"]) is estimator
    assert estimator.classes_.tolist() == ["cat", "dog", "owl"]
    assert estimator.predict_proba(x).shape == (2, 3)
    assert estimator.partial_fit(x[:1], ["owl"], classes=["dog", "owl", "cat"]) is estimator
    assert (estimator.n_iter_, estimator.t_) == (2, 3)
    before = estimator.training_digest_
    for labels, classes in [(["new"], None), (["owl"], ["cat", "owl"])]:
        with pytest.raises(ValueError):
            estimator.partial_fit(x[:1], labels, classes=classes)
        assert estimator.training_digest_ == before


def test_first_incremental_call_requires_declared_classes():
    x = np.ones((2, 2), np.float32)
    with pytest.raises(ValueError, match="classes"):
        MLPClassifier(device="cpu").partial_fit(x, [0, 1])


@pytest.mark.parametrize("solver", ["adam", "sgd"])
def test_full_fit_checkpoint_resumes_one_epoch_without_restarting(solver):
    x = np.array([[0, .5], [1, -1], [.25, 1]], np.float32)
    y = np.array([[1, -1], [.25, 0], [-.5, 1]], np.float32)
    settings = dict(hidden_layer_sizes=(3,), activation="tanh", batch_size=2, solver=solver,
                    learning_rate="invscaling", random_state=7, device="cpu")
    first = MLPRegressor(**settings, max_iter=1).fit(x, y)
    resumed = pickle.loads(pickle.dumps(first))
    assert (first.digest_, first.training_digest_) == (resumed.digest_, resumed.training_digest_)
    full = MLPRegressor(**settings, max_iter=2).fit(x, y)
    for model in (first, resumed):
        assert model.partial_fit(x, y) is model
        assert (model.n_iter_, model.t_) == (2, 6)
        np.testing.assert_array_equal(model.predict(x), full.predict(x))
        for a, b in zip(model.coefs_ + model.intercepts_, full.coefs_ + full.intercepts_):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(model.loss_curve_, full.loss_curve_)
    assert first.training_digest_ == resumed.training_digest_


def test_invalid_regression_output_and_feature_widths_preserve_fit():
    x = np.array([[1., 0.], [-1., 1.]], np.float32)
    model = MLPRegressor(hidden_layer_sizes=(2,), device="cpu", random_state=1)
    model.partial_fit(x, np.array([1., 2.], np.float32))
    before = model.training_digest_
    for source, target in [(x, x), (x[:, :1], x[:, 0])]:
        with pytest.raises(ValueError):
            model.partial_fit(source, target)
        assert model.training_digest_ == before


def test_checkpoint_predating_optimizer_state_can_predict_and_refit():
    x = np.array([[0, 1], [1, 0]], np.float32)
    y = np.array([0, 1], np.float32)
    model = MLPRegressor(hidden_layer_sizes=(2,), max_iter=1, random_state=1, device="cpu").fit(x, y)
    pred = model.predict(x)
    old_answer_digest = model.digest_
    del model.__dict__["_optimizer_state_"]
    np.testing.assert_array_equal(model.predict(x), pred)
    assert model.digest_ == old_answer_digest
    with pytest.raises(ValueError, match="no saved optimizer state"):
        model.partial_fit(x, y)
    model.fit(x, y).partial_fit(x[:1], y[:1])
    assert (model.n_iter_, model.t_) == (2, 3)
