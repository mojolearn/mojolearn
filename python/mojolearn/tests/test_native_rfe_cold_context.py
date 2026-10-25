"""RFE must initialize its own native input context in a fresh process."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = r'''
import json, sys
from pathlib import Path
order, selected, directory = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
if order == 'numpy-first':
    import numpy as np
    import mojolearn as ml
else:
    import mojolearn as ml
    import numpy as np
from mojolearn import _expansion_prep as prep
from mojolearn._verify_native_runtime import audit_native_call

# No classifier operation warms prep._CLS_CONTEXT before this public RFE fit.
x = np.arange(24, dtype=np.float32).reshape(8, 3) / 10
y = np.arange(8, dtype=np.float32)
model = prep.RFE(ml.Ridge(), n_features_to_select=selected)
before = prep._CLS_CONTEXT.get('ndarray') is np.ndarray
audit = audit_native_call(model.fit, x, y)
report = audit.report()
(directory / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
assert audit.assert_native() is model
assert prep._CLS_CONTEXT.get('ndarray') is np.ndarray
indices = model.get_support(indices=True)
assert len(indices) == selected
np.testing.assert_array_equal(np.asarray(model.transform(x)), x[:, indices])
assert type(model.__dict__['estimator_']) is ml.Ridge
(directory / 'result.json').write_text(json.dumps({
    'passed': True, 'order': order, 'selected': selected,
    'select_numpy_connected_before_fit': before, 'strict_native_fit': report['passed'],
    'package': ml.__file__}, indent=2) + '\n')
'''


@pytest.mark.parametrize('order', ['numpy-first', 'package-first'])
@pytest.mark.parametrize('selected', [3, 2])
def test_rfe_ridge_first_operation_connects_its_input_context(order, selected, tmp_path):
    # Keep child logs, including successful traces, for the release controller.
    home = Path(os.environ.get('MOJOLEARN_TEST_EVIDENCE', str(tmp_path)))
    output = home / ('rfe-cold-' + order + '-' + str(selected))
    output.mkdir(parents=True, exist_ok=True)
    result = subprocess.run([sys.executable, '-c', SCRIPT, order, str(selected), str(output)],
                            capture_output=True, text=True, timeout=120)
    (output / 'stdout.log').write_text(result.stdout)
    (output / 'stderr.log').write_text(result.stderr)
    assert result.returncode == 0, result.stderr + '\n' + result.stdout
    report = json.loads((output / 'result.json').read_text())
    assert report['passed'] and report['strict_native_fit'], report
