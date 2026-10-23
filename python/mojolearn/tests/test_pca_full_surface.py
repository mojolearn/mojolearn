"""Authored additive full-PCA exposure gates; root executes, never agents.

The default cases check the native owner's refusals. Numerical case requires explicit
remote CUDA/HIP opt-in and does not confer cross-vendor certification.
"""
import os
import sys

import numpy as np
import pytest

from mojolearn import PCA, _backend


def test_full_wide_matrix_refuses_outside_identical():
    # The dense route needs a tall matrix outside IDENTICAL (DEVIATION 593);
    # the native owner refuses it before any kernel runs.
    try:
        model = PCA(2, svd_solver='full', numeric_mode='fast')
        model.fit(np.ones((8, 3), dtype=np.float32))
    except ImportError:
        pytest.skip('this install carries no fast estimators binding')
    with pytest.raises(NotImplementedError, match='at least as many samples'):
        PCA(2, svd_solver='full', numeric_mode='fast').fit(np.ones((3, 8), dtype=np.float32))


def test_full_component_bound_refused_before_the_fit():
    with pytest.raises(ValueError, match="cannot exceed min"):
        PCA(4, svd_solver="full", numeric_mode="identical").fit(np.ones((3, 8), dtype=np.float32))


@pytest.mark.skipif(
    os.environ.get('MOJOLEARN_PCA_FULL_GATE') != '1', reason='root-only numerical opt-in')
def test_full_remote_small_reference_and_identical_repeat():
    expected_vendor = os.environ.get('MOJOLEARN_EXPECT_VENDOR')
    if sys.platform != 'linux' or expected_vendor not in ('cuda', 'hip'):
        pytest.fail('full PCA gate requires explicit remote CUDA/HIP vendor')
    if _backend.vendor() != expected_vendor or _backend.numeric_mode() != 'identical':
        pytest.fail('full PCA native mode/vendor mismatch')
    x = np.array([[0, 1, 2], [2, 0, 1], [1, 3, -1], [-2, 1, 0],
                  [4, -1, 2], [0, -3, 1], [3, 2, 4], [-1, 0, -2]], dtype=np.float32)
    original = x.tobytes()
    a = PCA(3, svd_solver='full').fit(x)
    b = PCA(3, svd_solver='full').fit(x)
    centered = x.astype(np.float64) - x.astype(np.float64).mean(axis=0)
    _, singular, _ = np.linalg.svd(centered, full_matrices=False)
    np.testing.assert_allclose(a.singular_values_, singular, rtol=2e-4, atol=2e-5)
    np.testing.assert_allclose(a.explained_variance_, singular**2 / 7, rtol=4e-4, atol=2e-5)
    np.testing.assert_allclose(a.inverse_transform(a.transform(x)), x, rtol=2e-4, atol=2e-4)
    for name in ('components_', 'mean_', 'explained_variance_',
                 'explained_variance_ratio_', 'singular_values_'):
        assert getattr(a, name).tobytes() == getattr(b, name).tobytes()
    assert x.tobytes() == original


@pytest.mark.parametrize("kwargs", [dict(svd_solver="arpack"), dict(tol=1e-4)])
def test_pca_refuses_arpack_by_name(kwargs):
    """svd_solver='arpack' is a third algorithm (implicitly restarted
    Lanczos); running an exact arm under its name would be a silent
    substitution, so it raises before any binding is touched, and so does
    its tolerance (merge review, 2026-09-27)."""
    x = np.arange(24, dtype=np.float32).reshape(8, 3)
    with pytest.raises(NotImplementedError, match="arpack|ARPACK"):
        PCA(2, **kwargs).fit(x)


@pytest.mark.parametrize("kwargs", [dict(algorithm="arpack"), dict(tol=1e-4)])
def test_truncated_svd_refuses_arpack_by_name(kwargs):
    from mojolearn import TruncatedSVD
    x = np.arange(24, dtype=np.float32).reshape(8, 3)
    with pytest.raises(NotImplementedError, match="arpack|ARPACK"):
        TruncatedSVD(2, **kwargs).fit(x)
