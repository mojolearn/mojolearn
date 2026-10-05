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
