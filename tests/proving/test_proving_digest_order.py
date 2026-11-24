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
