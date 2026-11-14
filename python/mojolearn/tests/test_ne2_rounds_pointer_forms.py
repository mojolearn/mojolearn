# SPDX-License-Identifier: Apache-2.0
"""Lane ne2-b-rounds (N047, N052, N053, N054, N059, N065, N078): the pointer
forms and the per-block / per-operation resolutions return what the old
routes returned. Each check compares against a reference that does not
depend on the changed route (NumPy indexing, the batch-invariance handle,
an enclosing resident scope), so it holds on every vendor and tier.

The index, bootstrap and permutation batch invariants (N053, N078) run
natively in checks/native_tests/core_ne2_rounds_{cpu,gpu}.mojo.
"""
import numpy as np
import pytest

import mojolearn
from mojolearn import resample as rs


def _np(a):
    return np.asarray(a)


# ----------------------------------------------------------- N078 refusal
def _x(n=257, seed=0, d=1):
    rng = np.random.default_rng(seed)
    x = (rng.random((n, d), dtype=np.float32) * 2.0 - 1.0).astype(np.float32)
    return x[:, 0].copy() if d == 1 else x


def test_permutation_refuses_nonfinite_in_second_sample_by_pooled_position():
    x, y = _x(10, 1), _x(10, 2)
    y[3] = np.nan
    with pytest.raises(Exception, match="position 13"):
        rs.permutation_test(x, y, n_resamples=50, random_state=0)


# -------------------------------------------------------------- N052 rows
def _rows_of(out_index):
    return _np(out_index).astype(np.int64)


@pytest.mark.parametrize("make", [
    lambda: np.arange(600, dtype=np.int64).reshape(200, 3),
    lambda: np.arange(2400, dtype=np.float64).reshape(200, 3, 4),
    lambda: np.arange(1200, dtype=np.int16).reshape(400, 3)[::2],      # strided rows, contiguous items
    lambda: np.asfortranarray(np.arange(600, dtype=np.int32).reshape(200, 3)),  # non-contiguous items
    lambda: np.array(["a%d" % i for i in range(200)]),
])
@pytest.mark.parametrize("replace", [True, False])
def test_resample_rows_equal_numpy_take(make, replace):
    x = make()
    marker = np.arange(len(x), dtype=np.int64)
    out, idx = rs.resample(x, marker, replace=replace, n_samples=150, random_state=11)
    rows = _rows_of(idx)
    assert np.array_equal(_np(out), x[rows])
    assert _np(out).dtype == x.dtype


def test_resample_sequence_builds_list():
    seq = [("item", i) for i in range(50)]
    marker = np.arange(50, dtype=np.int64)
    out, idx = rs.resample(seq, marker, n_samples=40, random_state=2)
    assert type(out) is list
    rows = _rows_of(idx)
    assert out == [seq[int(r)] for r in rows]
    assert all(out[i] is seq[int(r)] for i, r in enumerate(rows))


# ------------------------------------------------- N054, N047, N065 pandas
pd = pytest.importorskip("pandas")


def _frame(n=120):
    return pd.DataFrame({
        "i8": np.arange(n, dtype=np.int8),
        "i32": np.arange(n, dtype=np.int32) * 3,
        "u16": np.arange(n, dtype=np.uint16),
        "f32": np.linspace(-1, 1, n, dtype=np.float32),
        "f64": np.linspace(0, 5, n),
        "b": (np.arange(n) % 3 == 0),
    })


def test_resample_frame_mixed_dtypes_equal_iloc():
    df = _frame()
    marker = np.arange(len(df), dtype=np.int64)
    out, idx = rs.resample(df, marker, n_samples=90, random_state=4)
    rows = _rows_of(idx)
    expect = df.iloc[rows].to_numpy()
    got = _np(out)
    assert got.dtype == expect.dtype
    assert np.array_equal(got.view(np.uint8), np.ascontiguousarray(expect).view(np.uint8))


def test_resample_frame_twice_reuses_numpy_table():
    df = _frame(60)
    a = _np(rs.resample(df, n_samples=30, random_state=1))
    b = _np(rs.resample(df, n_samples=30, random_state=1))
    assert np.array_equal(a, b)


def test_resample_tz_period_interval_cells():
    n = 40
    s_tz = pd.Series(pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"))
    s_tz.iloc[5] = pd.NaT
    s_td = pd.Series(pd.to_timedelta(np.arange(n), unit="s"))
    s_p = pd.Series(pd.period_range("2024-01", periods=n, freq="M"))
    s_p.iloc[7] = pd.NaT
    s_iv = pd.Series(pd.arrays.IntervalArray.from_breaks(np.arange(n + 1)))
    marker = np.arange(n, dtype=np.int64)
    for s in (s_tz, s_td, s_p, s_iv):
        out, idx = rs.resample(s, marker, n_samples=n, random_state=8, replace=False)
        rows = _rows_of(idx)
        expect = s.iloc[rows].to_numpy(dtype=object) if s.dtype == object or str(s.dtype).startswith(("period", "interval", "datetime64[ns,")) else s.iloc[rows].to_numpy()
        got = _np(out)
        assert len(got) == n
        for g, e in zip(got, expect):
            assert (pd.isna(g) and pd.isna(e)) or g == e


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
