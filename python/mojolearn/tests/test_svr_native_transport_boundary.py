# SPDX-License-Identifier: Apache-2.0
"""SVR publication, retained storage and exact compiled-provider boundaries."""
import gc
import pickle

import numpy as np
import pytest

from mojolearn import SVR
from mojolearn._array import Array
from mojolearn._buffer import frombytes

__mojolearn_user_boundary__ = True


def _points():
    return np.array([[-1, 0], [1, 0]], dtype=np.float32)


def _model():
    return SVR(kernel="linear", C=2, epsilon=0.1, tol=1e-6,
               max_iter=10000, numeric_mode="identical")


def test_svr_public_prefixes_survive_native_fit_plan_release_and_pickle():
    model = _model().fit(_points(), [-1, 1])
    assert model.n_support_ == 2
    assert model.n_features_in_ == 2
    assert isinstance(model.n_iter_, int) and model.n_iter_ > 0
    support = np.asarray(model.support_)
    dual = np.asarray(model.dual_coef_)
    vectors = np.asarray(model.support_vectors_)
    assert support.dtype == np.dtype("int32") and support.shape == (2,)
    assert dual.dtype == np.dtype("float32") and dual.shape == (1, 2)
    assert vectors.dtype == np.dtype("float32") and vectors.shape == (2, 2)
    restored = pickle.loads(pickle.dumps(model))
    del model
    gc.collect()
    np.testing.assert_array_equal(support, [0, 1])
    np.testing.assert_array_equal(vectors, _points())
    np.testing.assert_allclose(dual, [[-0.45, 0.45]], rtol=0, atol=2e-4)
    np.testing.assert_allclose(restored.predict(_points()), [-0.9, 0.9], rtol=0, atol=2e-4)


def test_svr_replaced_public_mirror_invalidates_prepared_lease():
    model = _model().fit(_points(), [-1, 1])
    first = np.asarray(model.predict(_points())).copy()
    replacement = np.asarray(model.dual_coef_).copy() * np.float32(0.5)
    model.dual_coef_ = frombytes(replacement.tobytes(), "<f4", replacement.shape)
    gc.collect()
    np.testing.assert_allclose(model.predict(_points()), first * 0.5, rtol=0, atol=2e-4)
    np.testing.assert_allclose(model.predict(_points()), first * 0.5, rtol=0, atol=2e-4)


def test_svr_refit_drops_previous_prepared_generation():
    model = _model().fit(_points(), [-1, 1])
    old = np.asarray(model.predict(_points())).copy()
    model.fit(_points(), [-2, 2])
    np.testing.assert_allclose(old, [-0.9, 0.9], rtol=0, atol=2e-4)
    np.testing.assert_allclose(model.predict(_points()), [-1.9, 1.9], rtol=0, atol=2e-4)


def test_svr_bad_target_keeps_early_input_publication_and_no_fit_outputs():
    model = _model()
    with pytest.raises(ValueError, match="y|length|rows"):
        model.fit(_points(), [1])
    assert "input_copied_" in vars(model)
    assert "dual_coef_" not in vars(model)
    assert "support_" not in vars(model)
    assert "n_support_" not in vars(model)


@pytest.mark.parametrize("operation", ["fit", "predict"])
def test_svr_rebound_native_factory_is_refused_without_python_call(monkeypatch, operation):
    model = _model()
    if operation == "predict":
        model.fit(_points(), [-1, 1])
    calls = []

    def hook():
        calls.append(True)
        raise AssertionError("Python SVR factory invoked")

    monkeypatch.setattr(model._bind(), "svr_native_api_v1", hook)
    with pytest.raises(TypeError, match="compiled extension entry"):
        if operation == "fit":
            model.fit(_points(), [-1, 1])
        else:
            model.predict(_points())
    assert calls == []


@pytest.mark.parametrize("name", ["_base", "_readonly", "__setattr__"])
def test_svr_storage_protocol_rebinding_never_invokes_python_hook(monkeypatch, name):
    model = _model().fit(_points(), [-1, 1])
    calls = []

    def hook(*args):
        calls.append(args)
        raise AssertionError("Python Array storage hook invoked")

    monkeypatch.setattr(Array, name, hook if name == "__setattr__" else property(hook, hook))
    with pytest.raises(Exception, match="native|storage|layout|descriptor"):
        model.predict(_points())
    assert calls == []
