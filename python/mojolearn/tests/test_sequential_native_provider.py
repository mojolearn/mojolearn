# SPDX-License-Identifier: Apache-2.0
"""Rebuilt-candidate public proof: shared native whole-model dispatch.

This is intentionally not run by source-only lane checks. Root runs it on the
supplied compiled candidate. The packed ABI oracle executes the same native
arithmetic; cross-freeze bit comparison is a separate integration check.
"""
from types import BuiltinFunctionType

import numpy as np
import pytest

from mojolearn import _backend
from mojolearn._x_sequence_model import _SEQUENTIAL_CONTEXT
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests.test_sequential_native_execution import (
    build, raw_fit, same_state, assert_prediction,
)


@pytest.mark.parametrize('loss,width', [('mse', 1), ('cross_entropy', 3), ('bce_with_logits', 1)])
def test_public_whole_model_dispatch_ignores_poisoned_legacy_entries(monkeypatch, loss, width):
    entries = _SEQUENTIAL_CONTEXT['bindings'][_backend.default_mode()]
    assert isinstance(entries['model_native_api_v1'], BuiltinFunctionType)
    x = np.linspace(-1, 1, 36, dtype='f4').reshape(12, 3)
    y = x[:, 0].copy() if loss == 'mse' else (
        (np.arange(12) % 3).astype('i8') if width == 3 else (np.arange(12) % 2).astype('?'))
    expected = raw_fit(x, y, width, loss=loss, epochs=4)
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError('internal packed CPython whole-model dispatch')

    for name in ('model_n_params', 'model_fit', 'model_fit_labels', 'model_predict', 'model_predict_labels'):
        monkeypatch.setitem(entries, name, forbidden)
    model = build(width, loss)
    audit_native_call(model.fit, x.copy(), y.copy(), epochs=2, batch_size=4,
                      seed=(1 << 63) + 19).assert_native()
    audit_native_call(model.fit, x.copy(), y.copy(), epochs=2, resume=True,
                      seed=(1 << 63) + 19).assert_native()
    same_state(model, expected)
    for fresh in (x[:5].copy(), np.asfortranarray(x[::-2], dtype='f8')):
        assert_prediction(model, fresh, expected)
        if loss != 'mse':
            assert_prediction(model, fresh, expected, probabilities=True)
    assert not calls
