"""Installed ARIMA parameter copies must retain their native stride metadata.

Each case runs in a child because the original native lifetime defect crashes
the process. Expected parameter bits come from the fitted public params_ matrix,
independently sliced by the declared order, rather than an older release hash.
"""
import os
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize('case', ['ar', 'ma', 'seasonal_exog'])
def test_public_parameter_bits_and_owner_lifetime(tmp_path, case):
    code = r'''
import gc, json, sys
import numpy as np
import mojolearn as ml

case = sys.argv[1]
order = (0, 1, 1) if case == 'ma' else (1, 0, 0)
seasonal = (1, 0, 0, 4) if case == 'seasonal_exog' else (0, 0, 0, 0)
y = np.random.default_rng(123).normal(size=(4, 64)).astype(np.float32)
exog = np.random.default_rng(17).normal(size=(4, 64, 1)).astype(np.float32) if case == 'seasonal_exog' else None
model = ml.ARIMA(order=order, seasonal_order=seasonal, maxiter=3).fit(y, exog=exog)
params = np.asarray(model.params_).copy()
fields = object.__getattribute__(model, '__dict__')
intercept = fields['k_']
regressors = fields['n_exog_']
offset = 0
expected = {}
for name, width, flat in [('mu_', intercept, True), ('beta_', regressors, False),
                         ('ar_', order[0], False), ('ma_', order[2], False),
                         ('sar_', seasonal[0], False), ('sma_', seasonal[2], False),
                         ('sigma2_', 1, True)]:
    if name in ('mu_', 'beta_') and width == 0:
        try:
            getattr(model, name)
        except AttributeError:
            pass
        else:
            raise AssertionError(name + ' must be absent')
        continue
    value = params[:, offset:offset + width].copy()
    expected[name] = value.reshape(-1) if flat else value
    offset += width
assert offset == params.shape[1]
published = {}
for repeat in range(16):
    for name, want in expected.items():
        print('parameter', repeat, name, flush=True)
        actual = np.asarray(getattr(model, name))
        assert actual.shape == want.shape
        assert actual.dtype == np.float32
        np.testing.assert_array_equal(actual.view(np.uint32), want.view(np.uint32))
        assert not np.shares_memory(actual, np.asarray(model.params_))
        published[name] = actual
np.savez(sys.argv[2], params=params, **published)
del model, fields
gc.collect()
for name, actual in published.items():
    np.testing.assert_array_equal(actual.view(np.uint32), expected[name].view(np.uint32))
print(json.dumps({'case': case, 'parameters': len(expected), 'repeats': 16, 'passed': True}))
'''
    env = dict(os.environ, PYTHONFAULTHANDLER='1')
    env.pop('PYTHONPATH', None)
    result = subprocess.run([sys.executable, '-c', textwrap.dedent(code), case, str(tmp_path / 'parameters.npz')],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=90)
    (tmp_path / 'child.stdout').write_text(result.stdout)
    (tmp_path / 'child.stderr').write_text(result.stderr)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
