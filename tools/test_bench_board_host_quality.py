import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('host_quality', Path(__file__).with_name('bench_board_host_quality.py'))
quality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality)


def test_float_error_is_reported_not_claimed_identical():
    assert quality.compare({'y': np.array([2., 4.])}, {'y': np.array([1., 2.])}) == {'relative_error_vs_own_host': 1.}
    assert quality.compare({'y': np.array([1., 2.])}, {'y': np.array([1., 2.])}) == {'relative_error_vs_own_host': 0.}


@pytest.mark.parametrize('actual,reference', [
    ({}, {}), ({'x': np.array([1])}, {'y': np.array([1])}),
    ({'x': np.array([1])}, {'x': np.array([2])}),
    ({'x': np.array([np.nan])}, {'x': np.array([1.])}),
    ({'x': np.array([1.])}, {'x': np.array([np.inf])}),
    ({'x': np.ones(2)}, {'x': np.ones(3)}),
])
def test_invalid_outputs_fail(actual, reference):
    with pytest.raises(ValueError):
        quality.compare(actual, reference)


def test_existing_quality_errors_are_not_replaced(tmp_path):
    q = {'ours-fast': {'error': 'real failure'}}
    assert quality.enrich('adam', {}, {'ours-fast': {}}, q, 'not-a-python', tmp_path, 1) is None
    assert q == {'ours-fast': {'error': 'real failure'}}


def test_host_binding_private_alias_is_hashed(tmp_path):
    from types import SimpleNamespace
    from bench_board_host_quality import host_binding_artifacts
    path = tmp_path / '_mojolearn_training_host.so'; path.write_bytes(b'host fixture')
    rows = host_binding_artifacts({'mojolearn._host._mojolearn_training_host': SimpleNamespace(__file__=str(path))})
    assert len(rows) == 1 and rows[0]['file'] == str(path.resolve()) and len(rows[0]['sha256']) == 64


def test_actual_gpu_file_refused_even_with_host_alias(tmp_path):
    from types import SimpleNamespace
    from bench_board_host_quality import host_binding_artifacts
    path = tmp_path / '_mojolearn_training.so'; path.write_bytes(b'gpu fixture')
    with pytest.raises(RuntimeError, match='non-host'):
        host_binding_artifacts({'pretend_host': SimpleNamespace(__file__=str(path))})


def test_no_loaded_binding_refused():
    from bench_board_host_quality import host_binding_artifacts
    with pytest.raises(RuntimeError, match='no host binding'):
        host_binding_artifacts({})


def test_stateful_reference_matches_warmup_and_sample_history():
    class ClippingRunner:
        norm = 4096.
        calls = []
        def fit(self):
            self.calls.append('fit'); self.last = self.norm; self.norm = 1.
        def infer(self): self.calls.append('infer')
        def outputs(self): return {'norm': np.array([self.last])}
    runner = ClippingRunner()
    result = quality.run_reference(runner, 2)
    assert result['norm'].tolist() == [1.]
    assert runner.calls == ['fit', 'infer', 'fit', 'infer']


@pytest.mark.parametrize('count', [0, -1, True, 1.5])
def test_invalid_reference_history_refused(count):
    with pytest.raises(ValueError, match='positive integer'):
        quality.run_reference(None, count)
