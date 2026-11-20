# SPDX-License-Identifier: Apache-2.0
"""Validation labels of the model-building APIs at the Python boundary.

The validation, best-state restore and captured-program training values are
the native check checks/native_tests/training_autograd_usability.mojo."""
import hashlib

import numpy as np
import pytest

import mojolearn as ml


def digest(name, *values):
    h = hashlib.sha256()
    for value in values:
        h.update(np.ascontiguousarray(value).tobytes())
    print("USABILITY_DIGEST", name, h.hexdigest()[:16])


def data():
    rng = np.random.default_rng(41)
    x = rng.standard_normal((43, 3), dtype=np.float32)
    y = (x[:, :1] * 1.3 - x[:, 1:2] * 0.7).astype(np.float32)
    return x, y


def test_validation_labels_use_training_vocabulary():
    x, _ = data()
    y = np.where(x[:, 0] > 0, 40000000003, -90000000007).astype(np.int64)
    # Validation deliberately contains only one of the training classes.
    xv, yv = x[y > 0], y[y > 0]
    m = ml.Sequential().linear(2).compile(loss="cross_entropy", lr=0.03)
    m.fit(x, y, epochs=3, batch_size=16, validation_data=(xv, yv))
    assert len(m.validation_loss_curve_) == 3
    expected = -np.log(m.predict_proba(xv)[:, 1]).mean()
    np.testing.assert_allclose(m.validation_loss_curve_[-1], expected, rtol=2e-5)
    with pytest.raises(Exception, match="unknown class"):
        m.fit(x, y, epochs=1, validation_data=(xv, np.full_like(yv, 123)))
    digest("validation_class_codes", m.params_)
