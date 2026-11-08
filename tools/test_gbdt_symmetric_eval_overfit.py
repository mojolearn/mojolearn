# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""gbdt-symmetric-eval: the Iter detector fires and ShrinkToBestIteration cuts.

At the 0.8.66 freeze the lane refused `ties` (the detector never fired) and
`denormal` (the shrink never cut) on every column, because the fixture labels
are a deterministic rule of the features and the held-out curve only turned
by luck. The lane's od and sh fits now score held-out labels with every fifth
label flipped. The first test is source only; the second runs the lane's own
recipe on the two fixtures that refused (three GBDT fits of 4000 rows each,
on whatever backend `mojolearn` loads)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import identity_break as identity  # noqa: E402


def test_noisy_labels_flip_every_fifth_row_and_nothing_else():
    y = (np.arange(23) % 2).astype(np.int32)
    out = identity.gbdt_eval_noisy_labels(y)
    assert out.dtype == y.dtype and out.shape == y.shape
    flipped = np.flatnonzero(out != y)
    assert flipped.tolist() == list(range(0, 23, identity.GBDT_EVAL_FLIP_EVERY))
    assert np.array_equal(out[flipped], 1 - y[flipped])
    assert np.array_equal(y, (np.arange(23) % 2).astype(np.int32)), "input was modified"


def test_lane_revision_is_not_a_size_and_says_why():
    rev = identity.LANE_REVISIONS["gbdt-symmetric-eval"]
    assert rev == "noisy-eval-labels-20261108-1"
    assert "every fifth label flipped" in identity.NON_SIZE_REVISIONS["gbdt-symmetric-eval"]


@pytest.mark.parametrize("kind", ["ties", "denormal"])
def test_detector_fires_and_shrink_cuts_on_the_fixtures_that_refused(kind):
    ml = pytest.importorskip("mojolearn")
    X, yc, yr = identity.fixture(kind)
    Xh = identity.heldout(kind)
    # The lane raises RuntimeError when the detector does not fire or the
    # shrink does not cut; a pass returns its parts.
    fit = identity.LANES["gbdt-symmetric-eval"](ml, X, yc, yr, Xh.copy())
    assert "od_stopped" in fit and "shrunk_predict" in fit
    # The lane's assert, restated against the returned model: `sh` is the
    # estimator the lane hands back.
    sh = fit.est
    curve = np.asarray(sh.test_loss_curve_, dtype=np.float64)
    cut = int(np.argmin(curve)) + 1
    assert 0 < cut < len(curve)
