"""Required saved-model evidence cannot silently become a no-save declaration."""
from types import SimpleNamespace

import numpy as np
import pytest

import identity_break as identity


@pytest.mark.parametrize('lane', tuple(identity.REQUIRED_NUMERIC_PARTS))
@pytest.mark.parametrize('absence', ['missing-methods', 'not-implemented', 'explicit-na'])
def test_required_model_failure_is_recorded_instead_of_na(monkeypatch, lane, absence):
    monkeypatch.delenv(identity.HOST_INFER_ENV, raising=False)

    class UnsupportedSave:
        def save(self, path):
            raise NotImplementedError('serializer unavailable')
        @classmethod
        def load(cls, path):
            return cls()

    estimator = UnsupportedSave() if absence == 'not-implemented' else SimpleNamespace()
    outputs = np.array([1, 2, 3], dtype=np.float32)
    fit = identity._fit({'state': identity._h(outputs)}, estimator,
                        lambda _: (outputs,),
                        model_na='n/a:no-save' if absence == 'explicit-na' else None)
    infer, model, reload, error = identity._probe_fit(fit, lane)
    assert infer == identity._h(outputs)
    assert model is None and reload is None
    assert 'requires saved-model bytes' in error
    assert identity._column_verdict([model]) == 'REFUSED'


def test_function_lane_retains_its_declared_absence():
    fit = identity._fit({'value': 'a' * 16})
    assert identity._probe_fit(fit, 'function-only') == (
        'n/a:function', 'n/a:no-save', None, None)
