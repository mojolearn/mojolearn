# SPDX-License-Identifier: Apache-2.0
"""IsolationForest public boundary: the caller's buffers, admission and returned dtypes."""
import os

import numpy as np

# These are public-caller contracts. Public wrappers tell a user call from a library-internal one by the
# caller's module (providers/python/estimator_lifecycle_pyapi_unpack.mojo:16 estimator_outside): this module
# lives in mojolearn.tests, so without the marker its calls count as internal and IsolationForest returns its
# internal float32 scores instead of the public float64 ones (bindings/iforest_public_lifecycle_pack.mojo:190).
__mojolearn_user_boundary__ = True
import pytest

os.environ.setdefault("MOJOLEARN_NUMERIC_MODE", "identical")

import mojolearn as ml  # noqa: E402


# ---------------------------------------------------------------- moved from test_c3_c5_no_copies.py (vdv-pytest-a)
# 0.8.66 regression C3: IsolationForest borrows the caller's float32 matrix on a binary that keeps the forest
# resident and keeps its own copy on one that refits per scoring call; it never writes the caller's array.


def _c3_x(rows=64, cols=6, seed=3):
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(rng.standard_normal((rows, cols)).astype(np.float32))


def test_iforest_fit_copies_and_never_changes_the_caller_matrix():
    X = _c3_x(rows=256)
    reference = X.copy()
    a = ml.IsolationForest(n_estimators=8, random_state=5).fit(X)
    b = ml.IsolationForest(n_estimators=8, random_state=5).fit(reference.copy())
    assert X.flags.writeable and X.tobytes() == reference.tobytes()
    sa = np.asarray(a.score_samples(reference)).tobytes()
    assert sa == np.asarray(b.score_samples(reference)).tobytes()
    X[:] = 0.0  # scoring reads the fit's own copy of the training matrix
    assert np.asarray(a.score_samples(reference)).tobytes() == sa


def test_iforest_public_scores_are_the_float32_scores_widened():
    # Lane restore-iforest-kde: the device epilogue writes the float64
    # scores itself (IDN_IF_SCORES_F64); they are float32 values widened
    # exactly, as the door's cast wrote them, and decision_function is
    # score_samples - offset_ rounded once in float32.
    X = _c3_x(rows=256)
    a = ml.IsolationForest(n_estimators=8, random_state=5).fit(X)
    s = np.asarray(a.score_samples(X))
    d = np.asarray(a.decision_function(X))
    assert s.dtype == np.float64 and d.dtype == np.float64
    assert s.tobytes() == s.astype(np.float32).astype(np.float64).tobytes()
    expected = (s.astype(np.float32) - np.float32(a.offset_)).astype(np.float64)
    assert d.tobytes() == expected.tobytes()


@pytest.mark.parametrize("form", ["float64", "strided", "fortran"])
def test_iforest_fit_keeps_its_cast_copy_and_matches_the_float32_fit(form):
    # Lane perf-iforest: a non-float32-C X is cast once into storage the fit
    # owns and kept as the training matrix (no zero-filled store, no second
    # copy); the forest and the scores equal the float32 C fit's.
    reference = _c3_x(rows=256)
    if form == "float64":
        X = reference.astype(np.float64)
    elif form == "strided":
        X = np.repeat(reference, 2, axis=1)[:, ::2]
    else:
        X = np.asfortranarray(reference)
    a = ml.IsolationForest(n_estimators=8, random_state=5).fit(X)
    b = ml.IsolationForest(n_estimators=8, random_state=5).fit(reference.copy())
    assert type(a.__dict__["_x"]) is type(b.__dict__["_x"])
    assert np.asarray(a.__dict__["_x"]).tobytes() == reference.tobytes()
    sa = np.asarray(a.score_samples(reference)).tobytes()
    assert sa == np.asarray(b.score_samples(reference)).tobytes()
    X[...] = 0.0  # the fit kept its own cast copy
    assert np.asarray(a.score_samples(reference)).tobytes() == sa
    assert np.asarray(a.__dict__["_x"]).tobytes() == reference.tobytes()
