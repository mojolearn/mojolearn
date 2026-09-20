"""Required saved-model evidence cannot silently become a no-save declaration."""
from types import SimpleNamespace, ModuleType
import gc
import sys

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


@pytest.mark.parametrize('fail_forward', [False, True])
def test_parallel_causal_batch_closes_worker_without_garbage_collection(monkeypatch, fail_forward):
    instances = []

    class ParallelModel:
        @classmethod
        def load(cls, root, layer_devices):
            model = cls()
            model.closed = 0
            model.cycle = model
            instances.append(model)
            return model

        def forward(self, rows):
            assert not self.closed
            if fail_forward:
                raise ValueError('forward failed')
            return np.asarray(rows, dtype=np.float32)

        def close(self):
            self.closed += 1

    package = ModuleType('mojolearn')
    package._causal_lm_fixtures = SimpleNamespace(
        family_fixture=lambda *args: ({}, {}),
        _write_checkpoint=lambda root, *args: root)
    models = ModuleType('mojolearn.models')
    models.ParallelCausalLM = ParallelModel
    monkeypatch.setitem(sys.modules, 'mojolearn', package)
    monkeypatch.setitem(sys.modules, 'mojolearn.models', models)
    monkeypatch.setattr(identity, '_BATCH_CLOSE', [])
    monkeypatch.setattr(identity, '_par_devices', lambda: (0,))
    ml = SimpleNamespace(Array=SimpleNamespace(from_buffer=np.asarray),
        models=SimpleNamespace(causal_lm=SimpleNamespace(CausalLM=SimpleNamespace(
            load=lambda _: SimpleNamespace(plan=SimpleNamespace(n_layers=2))))))
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        value, error = identity._probe_batch(SimpleNamespace(est=None), 'par-causal-lm', ml,
            np.zeros((128, 4), dtype=np.float32), alone=2, sabotage=False)
        assert instances and instances[0].closed == 1
        assert identity._BATCH_CLOSE == []
        if fail_forward:
            assert value is None and 'forward failed' in error
        else:
            assert len(value) == 16 and error is None
    finally:
        if was_enabled:
            gc.enable()
