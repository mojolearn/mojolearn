# SPDX-License-Identifier: Apache-2.0
"""A fit leaves no state for the next model: the forest regressor twin digest after the category fits.

r6-proving (2026-11-23), L40S wheel 4519097d380a: `pytest tests/proving -p no:xdist` failed the forest regressor
twin digests (RandomForestRegressor, ExtraTreesRegressor; classifiers passed) only when
test_proving_categories.py had run first in the same process; each file alone passed. The order is replayed here in
one fresh process, so the regression holds whatever order pytest or xdist picks for this file.
"""
import subprocess
import sys
from pathlib import Path

import pytest

from _proving import identical

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(not identical(), reason="digest_ exists in identical mode only")


@pytest.mark.parametrize("twin_case", ["RandomForestRegressor", "ExtraTreesRegressor"])
def test_category_fits_then_a_forest_regressor_twin_has_the_models_digest(twin_case):
    twin = ("tests/proving/test_proving_digest.py::test_proving_host_twin_has_the_saving_models_digest"
            f"[{twin_case}-params{0 if twin_case == 'RandomForestRegressor' else 1}-r]")
    run = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:xdist", "-p", "no:cacheprovider",
                          "tests/proving/test_proving_categories.py", twin], cwd=ROOT, capture_output=True,
                         text=True, timeout=900)
    tail = "\n".join(run.stdout.splitlines()[-15:])
    assert run.returncode == 0, "the category fits left state for the next forest regressor:\n" + tail


_TWIN_AFTER_IMPORT = r'''
import sys, os, tempfile
import numpy as np
exec(sys.argv[1])
import mojolearn
r = np.random.default_rng(0); X = r.random((400, 6), dtype=np.float32); y = X @ np.arange(1, 7, dtype=np.float32)
bad = []
for name, target in (("RandomForestRegressor", y), ("RandomForestClassifier", (y > np.median(y)).astype(np.int32)),
                     ("ExtraTreesRegressor", y)):
    model = getattr(mojolearn, name)(n_estimators=5, random_state=0, device="cpu").fit(X, target)
    path = os.path.join(tempfile.mkdtemp(), name + ".npz"); model.save(path)
    if mojolearn.host_model(path).digest_ != model.digest_:
        bad.append(name)
# HostGBDT's family layout is taken with HostForest's, after the saved-model connection (lane w3-cleanup).
model = mojolearn.GradientBoosting(n_estimators=5, max_depth=3, random_state=0, device="cpu").fit(X, y)
path = os.path.join(tempfile.mkdtemp(), "GradientBoosting.npz"); model.save(path)
if mojolearn.host_model(path).digest_ != model.digest_:
    bad.append("GradientBoosting")
print("BAD " + ",".join(bad) if bad else "OK")
'''


@pytest.mark.parametrize("first", ["", "import mojolearn._classical_host", "import mojolearn._forest_host",
                                   "import mojolearn._gbdt_host"])
def test_the_twin_digest_does_not_depend_on_which_host_module_is_imported_first(first):
    # r6-proving (2026-11-23): `import mojolearn._classical_host` first (a CPU GBDT predict does it) took the
    # HostForest family layout before its finalizer was installed; the twin's digest then read other fields.
    run = subprocess.run([sys.executable, "-c", _TWIN_AFTER_IMPORT, first], cwd=ROOT, capture_output=True,
                         text=True, timeout=600)
    assert run.returncode == 0, run.stderr[-2000:]
    assert run.stdout.strip().splitlines()[-1] == "OK", f"{first or 'no prelude'}: {run.stdout.strip()}"
