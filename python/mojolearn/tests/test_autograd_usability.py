# SPDX-License-Identifier: Apache-2.0
"""Independent numerical and lifecycle checks of the model-building APIs."""
import hashlib

import numpy as np
import pytest

import mojolearn as ml
from mojolearn import ops


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


def regression():
    return ml.Sequential().linear(1).compile(loss="mse", optimizer="adam", lr=0.1)


def test_validation_loss_and_best_state_resume():
    x, y = data()
    xv, yv = x[:17], -y[:17]
    reference = []
    for count in range(1, 7):
        m = regression().fit(x, y, epochs=count, batch_size=16, shuffle=False, seed=3)
        reference.append((np.mean((m.predict(xv) - yv) ** 2), m.params_.copy()))
    best = int(np.argmin([v[0] for v in reference])) + 1
    m = regression().fit(x, y, epochs=6, batch_size=16, shuffle=False, seed=3,
                         validation_data=(xv, yv), restore_best_weights=True)
    assert m.epochs_run_ == 6
    assert m.n_iter_ == m.best_epoch_ == best
    np.testing.assert_allclose(m.validation_loss_curve_, [v[0] for v in reference], rtol=2e-5, atol=2e-6)
    np.testing.assert_array_equal(m.params_, reference[best - 1][1])
    m.fit(x, y, epochs=1, resume=True)
    expected = regression().fit(x, y, epochs=best + 1, batch_size=16, shuffle=False, seed=3)
    np.testing.assert_array_equal(m.params_, expected.params_)
    digest("validation_restore", m.params_)


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


def test_validation_early_stopping():
    x, y = data()
    m = regression().compile(loss="mse", lr=1e-6, n_iter_no_change=2, tol=10.0)
    m.fit(x, y, epochs=20, validation_data=(x, -y))
    assert m.epochs_run_ == 3
    assert len(m.validation_loss_curve_) == 3


def test_dynamic_program_shapes_and_recovery():
    rng = np.random.default_rng(19)
    w = rng.standard_normal((3, 2), dtype=np.float32)
    with ops.Session() as s:
        x = s.tensor(np.ones((2, 4, 3), np.float32))
        weight = s.tensor(w)
        out = s.linear(x, weight, activation="tanh")
        program = s.capture([x], [out])
        for sh in [(5, 7, 3), (1, 2, 3), (2, 4, 3)]:
            a = rng.standard_normal(sh, dtype=np.float32)
            result = program.run(a, dynamic_shapes=True)
            np.testing.assert_allclose(result, np.tanh(a @ w), rtol=2e-5, atol=2e-6)
            digest("dynamic_" + "_".join(map(str, sh)), result)
        with pytest.raises(Exception):
            program.run(np.ones((2, 4, 5), np.float32), dynamic_shapes=True)
        # A failed replay must restore the owning session and its scratch tape.
        result = program.run(np.ones((2, 4, 3), np.float32))
        np.testing.assert_allclose(result, np.tanh(np.ones((2, 4, 3), np.float32) @ w), atol=2e-6)


def test_program_fit_matches_eager_and_continues():
    x, y = data()
    w0 = np.array([[0.2], [-0.1], [0.3]], np.float32)
    with ops.Session(record=True) as s:
        w = s.tensor(w0, requires_grad=True)
        opt = s.optimizer("adam", lr=0.02)
        for _ in range(5):
            a, b = s.tensor(x), s.tensor(y)
            s.backward(s.mse(s.matmul(a, w), b))
            s.step(opt, [w])
            s.scheduler_step(opt)
            s.reset()
        expected = s.numpy(w)
    with ops.Session() as s:
        a, w = s.tensor(x), s.tensor(w0)
        model = s.capture([a], [s.matmul(a, w)])
        opt = s.optimizer("adam", lr=0.02)
        model.fit(x, targets=y, optimizer=opt, epochs=2, batch_size=len(x), shuffle=False)
        model.fit(x, targets=y, optimizer=opt, epochs=3, batch_size=len(x), shuffle=False)
        np.testing.assert_array_equal(s.numpy(w), expected)
        result = model.run(x[:9], dynamic_shapes=True)
        np.testing.assert_allclose(result, x[:9] @ expected, atol=2e-6)
        digest("program_adam", s.numpy(w), result)


def test_program_custom_loss_minibatches_and_frozen_constants():
    x, y = data()
    with ops.Session() as s:
        a, target = s.tensor(x), s.tensor(y)
        w = s.tensor(np.zeros((3, 1), np.float32))
        scale = s.tensor(np.array([[2]], np.float32))
        pred = s.mul(s.matmul(a, w), scale)
        model = s.capture([a, target], [s.mse(pred, target)])
        opt = s.optimizer("sgd", lr=0.03)
        model.fit(x, y, optimizer=opt, parameters=[w], loss="output", epochs=12, batch_size=16, seed=4)
        assert model.loss_curve_[-1] < model.loss_curve_[0] * 0.1
        np.testing.assert_array_equal(s.numpy(scale), [[2]])
        digest("program_custom_loss", s.numpy(w), model.loss_curve_)
