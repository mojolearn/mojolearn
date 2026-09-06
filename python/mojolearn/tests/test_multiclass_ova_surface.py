# SPDX-License-Identifier: Apache-2.0
"""OVA boundary and serialization checks; no GPU calls or native fitting."""
import ctypes

import numpy as np
import pytest

from mojolearn import _backend
from mojolearn.ensemble import GradientBoosting


class _Binding:
    def __init__(self, classes):
        self.classes = classes
        self.transforms = []

    def gbdt_fit(self, x, y, weights, flags, ex, ey, params, strings):
        self.params = list(params)
        self.strings = list(strings)
        self.weights = np.ctypeslib.as_array(
            (ctypes.c_float * int(params[2])).from_address(weights)
        ).copy()
        return "ova-model", 2, False, [0.7, 0.5, 0.4], []

    def gbdt_model_dim(self, text):
        return self.classes

    def gbdt_predict_multi(self, model, x, out, params):
        rows, transform = params
        self.transforms.append(transform)
        # Unequal, non-normalized probabilities reveal a softmax dispatch.
        values = np.arange(1, self.classes + 1, dtype=np.float32) / 8
        if transform == 0:
            values = values - 1
        elif transform != 2:
            raise AssertionError("OVA requested a non-sigmoid probability link")
        buffer = np.ctypeslib.as_array(
            (ctypes.c_float * (rows * self.classes)).from_address(out)
        )
        buffer[:] = np.tile(values, rows)
        return self.classes


@pytest.mark.parametrize("classes", [2, 3])
def test_ova_dimensions_link_weights_and_saved_mode(classes, monkeypatch, tmp_path):
    binding = _Binding(classes)
    selected_modes = []

    def bind(model, name):
        selected_modes.append(model.numeric_mode)
        return binding

    monkeypatch.setattr(GradientBoosting, "_bind", bind)
    X = np.arange(12, dtype=np.float32).reshape(6, 2)
    y = np.arange(6) % classes
    weights = np.arange(1, 7, dtype=np.float32) / 2
    class_weights = [float(k + 1) for k in range(classes)]
    model = GradientBoosting(loss="MultiClassOneVsAll", n_estimators=3,
                             class_weights=class_weights, numeric_mode="identical")
    model.fit(X, y, sample_weight=weights)
    assert binding.strings[0] == "MultiClassOneVsAll"
    assert binding.params[34:] == [classes, *class_weights]
    np.testing.assert_array_equal(binding.weights, weights)
    assert model.n_classes_ == model.approx_dim_ == classes
    raw, probability = model.predict(X), model.predict_proba(X)
    assert raw.shape == probability.shape == (6, classes)
    assert raw.dtype == probability.dtype == np.float32
    assert binding.transforms == [0, 2]
    assert not np.allclose(probability.sum(axis=1), 1)
    np.testing.assert_array_equal(model.predict_classes(X), np.full(6, classes - 1))
    path = tmp_path / "ova.npz"
    model.save(path)
    selected_modes.clear()
    monkeypatch.setattr(_backend, "default_mode", lambda: "fast")
    restored = GradientBoosting.load(path)
    assert restored.loss == "MultiClassOneVsAll"
    assert restored.numeric_mode == "identical"
    assert restored.n_classes_ == restored.approx_dim_ == classes
    np.testing.assert_array_equal(restored.predict(X).view(np.uint32), raw.view(np.uint32))
    np.testing.assert_array_equal(restored.predict_proba(X).view(np.uint32), probability.view(np.uint32))
    assert selected_modes and set(selected_modes) == {"identical"}


@pytest.mark.parametrize("labels", [
    [-1, 0, 1], [0, 0.5, 1], [0, np.nan, 1], [0, np.inf, 1], [0, 0, 0],
])
def test_unsafe_ova_labels_refused_before_binding(labels):
    model = GradientBoosting(loss="MultiClassOneVsAll", class_weights=[1, 2])
    model._bind = lambda name: pytest.fail("unsafe labels reached native binding")
    with pytest.raises(ValueError, match="class"):
        model.fit(np.zeros((3, 2), np.float32), labels)


@pytest.mark.parametrize("policy", ["Depthwise", "Lossguide"])
def test_ova_remains_symmetric_only(policy):
    with pytest.raises(NotImplementedError, match="optimization scheme"):
        GradientBoosting(loss="MultiClassOneVsAll", grow_policy=policy)
