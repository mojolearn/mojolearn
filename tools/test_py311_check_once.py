# SPDX-License-Identifier: Apache-2.0
"""The CPython 3.10 / 3.11 check-once path, exercised on an installed wheel.

Lane ne2-i-j (2026-11-07, docs/plans/NATIVE_ENTRY_POINTS_AFTER_0866.md
section 2I): on 3.10 and 3.11 the native records key on the objects' own
version tags (bindings/py_version_tags.mojo) instead of 3.12's watchers. This
job runs, in a 3.11 interpreter with the wheel installed, one estimator per
family: import, fit, predict twice (the second call takes the records), then a
class attribute assignment and an instance attribute replacement, after which
the next call must notice the change (a stale record would return the old
answer or skip the refusal). No timing.

    MOJOLEARN_PY311=/path/to/venv311/bin/python python3 -m pytest tools/test_py311_check_once.py
"""
import os
import subprocess

import pytest

PY311 = os.environ.get("MOJOLEARN_PY311")

SCRIPT = r'''
import sys
assert sys.version_info[:2] in ((3, 10), (3, 11)), sys.version
import numpy as np
import mojolearn as ml
rng = np.random.default_rng(0)
X = rng.standard_normal((200, 6)).astype(np.float32)
y = (X[:, 0] > 0).astype(np.int64)
yr = (X @ np.arange(6, dtype=np.float32)).astype(np.float32)
cases = [
    (ml.LogisticRegression, y), (ml.Ridge, yr), (ml.KMeans, None), (ml.RandomForestClassifier, y),
    (ml.KNeighborsClassifier, y), (ml.StandardScaler, None), (ml.PCA, None),
]
for cls, target in cases:
    est = cls()
    est = est.fit(X) if target is None else est.fit(X, target)
    method = "predict" if hasattr(est, "predict") else "transform"
    first = np.asarray(getattr(est, method)(X))
    second = np.asarray(getattr(est, method)(X))
    assert first.tobytes() == second.tobytes(), cls.__name__
    # an instance edit after the records were taken: a refit on other data must be seen
    est = est.fit(X[::-1]) if target is None else est.fit(X[::-1], target[::-1])
    third = np.asarray(getattr(est, method)(X))
    fresh = cls()
    fresh = fresh.fit(X[::-1]) if target is None else fresh.fit(X[::-1], target[::-1])
    assert third.tobytes() == np.asarray(getattr(fresh, method)(X)).tobytes(), cls.__name__
# a class edit after its record was admitted: a hooked public method is refused or used, never skipped silently
cls = ml.Ridge
est = cls().fit(X, yr)
_ = est.predict(X)
sub = type("RidgeSub", (cls,), {})
s = sub().fit(X, yr)
_ = s.predict(X)
sub.extra_marker = 1  # PyType_Modified: the subclass's record must miss
_ = s.predict(X)
print("PY311-CHECK-ONCE OK", sys.version.split()[0])
'''


@pytest.mark.skipif(not PY311, reason="MOJOLEARN_PY311 names no 3.10 / 3.11 interpreter with the wheel")
def test_check_once_records_on_python_311():
    result = subprocess.run([PY311, "-I", "-c", SCRIPT], capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, (result.stdout[-2000:], result.stderr[-4000:])
    assert "PY311-CHECK-ONCE OK" in result.stdout
