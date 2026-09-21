"""Routing contract for the FAST resident binary-class output boundary."""
import ctypes
from types import SimpleNamespace

import numpy as np

from mojolearn import GradientBoosting
from mojolearn._array import Array
from mojolearn._buffer import as_f32_c


def _model(mode):
    model = GradientBoosting(loss="Logloss", numeric_mode=mode)
    model.model_ = "model"
    model.n_features_in_ = 1
    model.approx_dim_ = 1
    return model


def test_fast_binary_predict_classes_uses_resident_codes(monkeypatch):
    calls = []

    def direct(handle, source, destination, params):
        calls.append((handle, tuple(params)))
        out = (ctypes.c_int64 * params[0]).from_address(destination)
        for i, value in enumerate((0, 1, 0)):
            out[i] = value
        return params[0]

    native = SimpleNamespace(gbdt_resident_binary_classes=direct)
    model = _model("fast")
    source = as_f32_c(np.ones((3, 1), np.float32), name="X")[0]
    monkeypatch.setattr(model, "numeric_mode_used", lambda: "fast")
    monkeypatch.setattr(model, "_bind", lambda *args: native)
    monkeypatch.setattr(model, "_resident_handle", lambda binding: 17)
    monkeypatch.setattr(model, "_check_fitted_layout", lambda X: (source, 3, True))
    np.testing.assert_array_equal(model.predict_classes(source), [0, 1, 0])
    assert calls == [(17, (3, 1))]


def test_reproducibility_tiers_and_multiclass_keep_probability_path(monkeypatch):
    model = _model("identical")
    monkeypatch.setattr(model, "numeric_mode_used", lambda: "identical")
    monkeypatch.setattr(
        model, "_bind",
        lambda *args: SimpleNamespace(
            gbdt_resident_binary_classes=lambda *args: (_ for _ in ()).throw(
                AssertionError("reproducibility tier reached FAST class boundary")
            )
        ),
    )
    monkeypatch.setattr(
        model, "predict_proba",
        lambda X: Array.from_list([[0.5, 0.5], [0.25, 0.75]], "<f8"),
    )
    np.testing.assert_array_equal(model.predict_classes([[0.0], [1.0]]), [0, 1])

    # FAST fuses narrow (at most three classes) symmetric multiclass output
    # on the resident model; wider multiclass keeps the probability path.
    model.loss = "MultiClass"
    model.n_classes_ = 4
    monkeypatch.setattr(model, "numeric_mode_used", lambda: "fast")
    monkeypatch.setattr(
        model, "predict_proba",
        lambda X: Array.from_list(
            [[0.4, 0.4, 0.1, 0.1], [0.1, 0.2, 0.6, 0.1]], "<f4"),
    )
    np.testing.assert_array_equal(model.predict_classes([[0.0], [1.0]]), [0, 2])
