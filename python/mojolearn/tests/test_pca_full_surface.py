"""Full-PCA refusals of the native owner.

The small 8x3 reference spectrum, reconstruction and input-ownership case runs
over the CPU/GPU PCA providers in checks/native_tests/pca_native_* (fixture
full-8x3, the same matrix and tolerances).
"""
import numpy as np
import pytest

from mojolearn import PCA


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
