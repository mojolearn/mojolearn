# SPDX-License-Identifier: Apache-2.0
"""Candidate-runtime metadata refresh tests; require rebuilt native bindings."""
import gc

import numpy as np
import pytest

from mojolearn import MLPRegressor, Sequential, ops
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests.test_sequential_native_execution import (
    assert_prediction, build, raw_fit, same_state,
)


def test_sequential_snapshot_refreshes_layout_dtype_rows_and_values(tmp_path):
    x = np.linspace(-1, 1, 36, dtype='f4').reshape(12, 3)
    y = (x[:, 0] - 0.5 * x[:, 1]).copy()
    model = build(1)
    expected = raw_fit(x, y, 1, epochs=2)
    assert audit_native_call(model.fit, x.copy(), y.copy(), epochs=2, batch_size=4,
                             seed=(1 << 63) + 19).assert_native() is model
    same_state(model, expected)
    checkpoint = tmp_path / 'metadata.ckpt'
    audit_native_call(model.save_checkpoint, checkpoint).assert_native()
    model = audit_native_call(Sequential.from_checkpoint, checkpoint).assert_native()
    same_state(model, expected)
    # Descriptor reuse ends at return: each prediction admits its current owner.
    for fresh in (x[:5].copy(), np.asfortranarray(x[:7], dtype='f8'), x[::-2],
                  np.repeat(x[:4], 2, axis=1)[:, ::2]):
        assert_prediction(model, fresh, expected)
    changed = x[:5].copy()
    changed[:] += 0.25
    assert_prediction(model, changed, expected)
    output = model.predict(changed.copy())
    retained = output.tobytes()
    del changed
    gc.collect()
    assert output.tobytes() == retained
    with pytest.raises(ValueError, match='feature'):
        model.predict(x.reshape(6, 6))
    # Refit the same model using a different owner, dtype, row count and targets.
    next_x = np.linspace(-0.5, 0.5, 27, dtype='f8').reshape(9, 3)
    next_y = next_x[:, 1].copy()
    next_expected = raw_fit(next_x, next_y, 1, epochs=2)
    audit_native_call(model.fit, next_x, next_y, epochs=2, batch_size=4,
                      seed=(1 << 63) + 19).assert_native()
    same_state(model, next_expected)
    assert_prediction(model, next_x[:3], next_expected)


def test_sequential_native_sample_shape_refresh_and_validation():
    x = np.linspace(-1, 1, 40, dtype='f4').reshape(10, 2, 2)
    y = x[:, 0, 0].copy()
    model = build(1, shaped=True)
    expected = raw_fit(x, y, 1, epochs=2, shaped=True)
    audit_native_call(model.fit, x, y, epochs=2, batch_size=4,
                      seed=(1 << 63) + 19).assert_native()
    same_state(model, expected)
    assert_prediction(model, x[:3].copy(), expected)
    # Same flattened width but different sample axes must still be rejected.
    with pytest.raises(ValueError, match='shape'):
        model.predict(x.reshape(10, 1, 4))
    with pytest.raises(ValueError, match='shape'):
        model.fit(x, y, epochs=1, validation_data=(x.reshape(10, 1, 4), y))
    assert_prediction(model, x[:2].copy(), expected)


def test_mlp_prediction_rereads_replaced_fitted_weights():
    x = np.linspace(-1, 1, 24, dtype='f4').reshape(8, 3)
    y = x[:, 0].copy()
    model = MLPRegressor(hidden_layer_sizes=(4,), max_iter=2, batch_size=4,
                         random_state=7, shuffle=False).fit(x, y)
    original = np.asarray(model.predict(x)).copy()
    # Public fitted arrays remain mutable/replacable data; their metadata is
    # operation-local, not cached solely under the estimator object's identity.
    replacement = [np.asarray(value).copy() for value in model.intercepts_]
    replacement[-1] += 1.0
    model.intercepts_ = replacement
    actual = np.asarray(audit_native_call(model.predict, x.copy()).assert_native())
    np.testing.assert_allclose(actual, original + 1.0, atol=2e-6, rtol=2e-6)
    assert np.asarray(model.predict(np.asfortranarray(x[:3], dtype='f8'))).shape == (3,)


def test_program_replay_refreshes_input_descriptors_and_keeps_outputs():
    x = np.arange(12, dtype='f4').reshape(4, 3) / 16
    w = np.arange(6, dtype='f4').reshape(3, 2) / 16
    with ops.Session() as session:
        a, b = session.tensor(x), session.tensor(w)
        output = session.linear(a, b, activation='relu')
        program = session.capture(inputs=[a, b], outputs=[output])
        session.reset()
        for current in (x.copy(), np.asfortranarray(x[::-1], dtype='f8'), x[:2].copy()):
            with ops.Session() as oracle:
                expected = oracle.numpy(oracle.linear(oracle.tensor(current), oracle.tensor(w), activation='relu'))
            got = audit_native_call(program.run, current, w.copy(), dynamic_shapes=True).assert_native()
            assert got.shape == expected.shape
            assert got.tobytes() == expected.tobytes()
        retained = got.tobytes()
    gc.collect()
    assert got.tobytes() == retained


def test_samba_dropout_metadata_refreshes_for_each_gradient_operation():
    from mojolearn.tests.test_samba_gradients_native import _model, _reference

    model = _model(['attention'], True)
    for shape, stream in (((1, 3), 11), ((2, 2), 19)):
        inputs = (np.arange(np.prod(shape), dtype='i4').reshape(shape) + 1)
        targets = inputs + 1
        expected_loss, expected_gradients = _reference(model, inputs, targets, None, stream, 3)
        rng_before = model.generator.state_dict()
        result = audit_native_call(model.loss_and_grads, inputs.copy(), targets.copy(),
                                   dropout_stream=stream, token_offset=3).assert_native()
        assert np.float32(result[0]).tobytes() == np.float32(expected_loss).tobytes()
        assert len(result[1]) == len(expected_gradients)
        for actual, expected in zip(result[1], expected_gradients):
            assert actual.tobytes() == expected
        assert model.generator.state_dict() == rng_before
