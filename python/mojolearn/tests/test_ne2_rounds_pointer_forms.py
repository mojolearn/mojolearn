# SPDX-License-Identifier: Apache-2.0
"""Lane ne2-b-rounds (N054, N059): a repeated frame resample and RFE inside
and outside an enclosing resident scope return the same rows and rankings.

The index, bootstrap and permutation batch invariants (N053, N078) run
natively in checks/native_tests/core_ne2_rounds_{cpu,gpu}.mojo; the resample
row, dtype, container and pandas-cell checks and the permutation_test
refusal are in test_resample_boundary.py.
"""
import numpy as np
import pytest

import mojolearn
from mojolearn import resample as rs


def _np(a):
    return np.asarray(a)


# ------------------------------------------------------------- N054 pandas
# pandas is optional: only the frame test below skips without it; the RFE
# test in this file runs either way.
@pytest.fixture
def pd():
    return pytest.importorskip("pandas")


def _frame(pd, n=120):
    return pd.DataFrame({
        "i8": np.arange(n, dtype=np.int8),
        "i32": np.arange(n, dtype=np.int32) * 3,
        "u16": np.arange(n, dtype=np.uint16),
        "f32": np.linspace(-1, 1, n, dtype=np.float32),
        "f64": np.linspace(0, 5, n),
        "b": (np.arange(n) % 3 == 0),
    })


def test_resample_frame_twice_reuses_numpy_table(pd):
    df = _frame(pd, 60)
    a = _np(rs.resample(df, n_samples=30, random_state=1))
    b = _np(rs.resample(df, n_samples=30, random_state=1))
    assert np.array_equal(a, b)


# --------------------------------------------------------------- N059 RFE
def test_rfe_rounds_same_with_and_without_enclosing_scope():
    from mojolearn import _arena_io
    from mojolearn import _expansion_prep as prep
    rng = np.random.default_rng(0)
    x = rng.standard_normal((80, 12)).astype(np.float32)
    y = (x[:, 0] - 2 * x[:, 3] + 0.1 * rng.standard_normal(80)).astype(np.float32)
    a = prep.RFE(mojolearn.Ridge(), n_features_to_select=3, step=2).fit(x, y)
    assert _arena_io.resident.stats() is None  # the RFE scope closed
    with _arena_io.resident():
        b = prep.RFE(mojolearn.Ridge(), n_features_to_select=3, step=2).fit(x, y)
    assert list(a.support_) == list(b.support_)
    assert np.array_equal(_np(a.ranking_), _np(b.ranking_))
    assert np.array_equal(_np(a.transform(x)).view(np.uint32), _np(b.transform(x)).view(np.uint32))
