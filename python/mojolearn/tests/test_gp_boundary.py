# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Python API boundary tests of the gp family: refusals, exception types and words, returned types,
buffer admission and framework interop. Numerical results are held by the verify lanes."""

# moved from python/mojolearn/tests/test_parallel_gaussian_process.py
import numpy as np
import pytest

from mojolearn import parallel_gaussian_process as _pgp_pg
from mojolearn._gpc_impl import GaussianProcessClassifier as _PgpGPC


@pytest.fixture
def _pgp_identical(monkeypatch):
    monkeypatch.setattr('mojolearn._backend.default_mode', lambda: 'identical')
    yield


@pytest.mark.parametrize('labels', [[1, 1], [1], []])
def test_parallel_gpc_invalid_targets(_pgp_identical, labels):
    with pytest.raises(ValueError):
        _pgp_pg.fit_gaussian_process_classifier(_PgpGPC(), np.ones((2, 3), np.float32), labels)


def test_parallel_gpc_prediction_validation(_pgp_identical):
    with pytest.raises(ValueError, match='method'):
        _pgp_pg.predict_gaussian_process_classifier(_PgpGPC(), np.ones((2, 3), np.float32), method='score')
    with pytest.raises(ValueError, match='fit'):
        _pgp_pg.predict_gaussian_process_classifier(_PgpGPC(), np.ones((2, 3), np.float32))
