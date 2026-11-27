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


def test_class_vocabulary_keeps_large_integers_distinct_from_rounded_float_labels():
    x = np.ones((2, 2), np.float32)
    classes = np.array([2**53 + 3, 2**53 + 12], np.int64)
    model = MLPClassifier(hidden_layer_sizes=(2,), device="cpu", random_state=1)
    model.partial_fit(x, classes, classes=classes)
    before = model.training_digest_
    with pytest.raises(ValueError, match="not one of the fitted classes"):
        model.partial_fit(x[:1], np.array([2**53 + 4], np.float64))
    with pytest.raises(ValueError, match="classes must match"):
        model.partial_fit(x, classes, classes=classes.astype(np.float64))
    assert model.training_digest_ == before


@pytest.mark.parametrize("classes", [np.array([[0], [1]]), np.array([0., 1., 1.5])])
def test_class_declaration_uses_the_classifier_target_contract(classes):
    model = MLPClassifier(hidden_layer_sizes=(2,), device="cpu", random_state=1)
    x = np.ones((2, 2), np.float32)
    with pytest.raises(ValueError):
        model.partial_fit(x, [0, 1], classes=classes)
    assert "coefs_" not in vars(model)


def test_resume_refuses_checkpoint_conversion_hooks_and_corrupt_counters():
    class Tag:
        def __index__(self):
            raise AssertionError("checkpoint conversion hook ran")

        __int__ = __index__

    model = MLPRegressor(hidden_layer_sizes=(2,), device="cpu", random_state=1)
    x = np.eye(2, dtype=np.float32)
    y = x[:, 0]
    model.partial_fit(x, y)
    before = model.digest_
    state = vars(model)["_optimizer_state_"]
    vars(model)["_optimizer_state_"] = (*state[:3], Tag(), *state[4:])
    with pytest.raises(TypeError, match="exact integers"):
        model.partial_fit(x, y)
    for moments in (state[0].astype(np.float64), state[0][::-1], state[0].reshape((1, -1))):
        vars(model)["_optimizer_state_"] = (moments, *state[1:])
        with pytest.raises(ValueError, match="vector layout"):
            model.partial_fit(x, y)
    for index, bad in ((0, 0), (1, np.iinfo(np.int64).max), (2, -1)):
        progress = state[1].copy()
        progress[index] = bad
        vars(model)["_optimizer_state_"] = (state[0], progress, *state[2:])
        with pytest.raises(ValueError, match="checkpoint counters"):
            model.partial_fit(x, y)
    assert (model.n_iter_, model.t_, model.digest_) == (1, 2, before)
    vars(model)["_optimizer_state_"] = state
    model.partial_fit(x, y)
    assert (model.n_iter_, model.t_) == (2, 4)


def test_partial_fit_preserves_the_original_dataframe_column_contract():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(np.eye(2, dtype=np.float32), columns=["left", "right"])
    model = MLPRegressor(hidden_layer_sizes=(2,), random_state=2, device="cpu")
    model.partial_fit(frame, [0., 1.])
    with pytest.warns(UserWarning, match="feature names"):
        model.partial_fit(frame.to_numpy(), [0., 1.])
    assert list(model.feature_names_in_) == ["left", "right"]
    with pytest.raises(ValueError, match="feature names"):
        model.partial_fit(frame[["right", "left"]], [0., 1.])


def test_overlapping_partial_fit_preserves_every_accepted_epoch():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    x = np.arange(32, dtype=np.float32).reshape(16, 2) / 16
    y = x[:, 0]
    options = dict(hidden_layer_sizes=(3,), batch_size=2, random_state=7, device="cpu")
    model = MLPRegressor(**options).partial_fit(x, y)
    start = Barrier(2)

    def update():
        start.wait()
        try:
            model.partial_fit(x, y)
        except RuntimeError as error:
            assert "already training" in str(error)
            return 0
        return 1

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(update) for _ in range(2)]
        accepted = sum(f.result() for f in futures)
    assert accepted >= 1
    assert "_mojolearn_fitting" not in vars(model)
    serial = MLPRegressor(**options)
    for _ in range(accepted + 1):
        serial.partial_fit(x, y)
    assert (model.digest_, model.training_digest_) == (serial.digest_, serial.training_digest_)
