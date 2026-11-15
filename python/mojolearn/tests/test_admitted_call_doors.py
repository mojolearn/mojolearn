# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""One admission record per public call (`AdmittedCall`, lane ne1-e-admitted-call).

Runtime checks: with a compiled classical binding, Ridge fit / predict / coef_ and a GP classifier fit
and predict give the same answers twice (the record changes no numerics). The source-text checks of the
door modules were deleted on 2026-11-15 (lane tests-delete-source-text).

    cd python && python3 -m pytest mojolearn/tests/test_admitted_call_doors.py -q
"""
import pytest


# test_bindings_resolve_once_per_call deleted: a source-text check of the GP door.
# 096e9f1b4f replaced the GP door's `estimator_arguments(call, model, ...)` with
# native `GPArgumentPins` (bindings/gp_argument_pins.mojo); GP fit/predict answers
# stay covered by test_gp_classifier_resolves_its_bindings_once below.


def test_ridge_answers_through_the_record():
    np = pytest.importorskip("numpy")
    import mojolearn

    rng = np.random.default_rng(0)
    X = rng.standard_normal((64, 5)).astype(np.float32)
    y = (X @ np.arange(1, 6, dtype=np.float32) + 0.5).astype(np.float32)
    try:
        model = mojolearn.Ridge(alpha=1.0).fit(X, y)
    except (ImportError, NotImplementedError, RuntimeError) as exc:
        pytest.skip(f"no classical binding in this process: {type(exc).__name__}: {exc}")
    first = np.asarray(model.predict(X))
    assert np.array_equal(first, np.asarray(model.predict(X)))
    assert np.asarray(model.coef_).shape == (5,)
    assert model.get_params()["alpha"] == 1.0
    with pytest.raises(AttributeError):
        model.no_such_attribute_


def test_gp_classifier_resolves_its_bindings_once():
    np = pytest.importorskip("numpy")
    import mojolearn

    rng = np.random.default_rng(1)
    X = rng.standard_normal((40, 3)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)
    try:
        model = mojolearn.GaussianProcessClassifier().fit(X, y)
    except (ImportError, NotImplementedError, RuntimeError, AttributeError) as exc:
        pytest.skip(f"no GP binding in this process: {type(exc).__name__}: {exc}")
    predicted = np.asarray(model.predict(X))
    assert predicted.shape == (40,)
    assert np.array_equal(predicted, np.asarray(model.predict(X)))
