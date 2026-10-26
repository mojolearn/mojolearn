"""The sharded forest accepts native NumPy inputs in a fresh process."""
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("classifier", [True, False])
@pytest.mark.parametrize("late_numpy", [True, False])
def test_parallel_forest_first_call(classifier, late_numpy):
    # A fresh process is essential: another owner can populate the shared
    # native type table and hide a missing first-call connection.
    imports = (
        "import mojolearn as ml\nimport numpy as np"
        if late_numpy else "import numpy as np\nimport mojolearn as ml"
    )
    code = imports + f"""
from mojolearn.parallel_ensemble import fit_forest
X = np.arange(128, dtype=np.float32).reshape(32, 4) / 128
classifier = {classifier!r}
y = np.arange(32, dtype=np.int64) % 2 if classifier else X[:, 0].copy()
cls = ml.RandomForestClassifier if classifier else ml.RandomForestRegressor
kw = dict(n_estimators=4, max_depth=3, random_state=7)
parallel = fit_forest(cls(**kw), X, y, devices=(0,), trees_per_shard=2)
plain = cls(**kw).fit(X, y)
for method in (('predict', 'predict_proba') if classifier else ('predict',)):
    got = np.asarray(getattr(parallel, method)(X))
    expected = np.asarray(getattr(plain, method)(X))
    assert got.dtype == expected.dtype and got.shape == expected.shape
    assert got.tobytes() == expected.tobytes(), method
"""
    result = subprocess.run(
        [sys.executable, "-c", code], env=os.environ.copy(),
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
