# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Python API boundary tests of the resample family: refusals, exception types and words, returned types,
buffer admission and framework interop. Numerical results are held by the verify lanes."""

# moved from python/mojolearn/tests/test_ne2_rounds_pointer_forms.py
# Lane ne2-b-rounds (N047, N052, N054, N065, N078): resample's pointer forms
# return NumPy indexing's rows, dtypes and container types, pandas cells
# included, and permutation_test refuses a non-finite value by its pooled position.
import numpy as np
import pytest

from mojolearn import resample as rs


def _ne2r_x(n=257, seed=0, d=1):
    rng = np.random.default_rng(seed)
    x = (rng.random((n, d), dtype=np.float32) * 2.0 - 1.0).astype(np.float32)
    return x[:, 0].copy() if d == 1 else x


def _ne2r_rows_of(out_index):
    return np.asarray(out_index).astype(np.int64)


def test_permutation_refuses_nonfinite_in_second_sample_by_pooled_position():
    x, y = _ne2r_x(10, 1), _ne2r_x(10, 2)
    y[3] = np.nan
    with pytest.raises(Exception, match="position 13"):
        rs.permutation_test(x, y, n_resamples=50, random_state=0)


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
    rows = _ne2r_rows_of(idx)
    assert np.array_equal(np.asarray(out), x[rows])
    assert np.asarray(out).dtype == x.dtype


def test_resample_sequence_builds_list():
    seq = [("item", i) for i in range(50)]
    marker = np.arange(50, dtype=np.int64)
    out, idx = rs.resample(seq, marker, n_samples=40, random_state=2)
    assert type(out) is list
    rows = _ne2r_rows_of(idx)
    assert out == [seq[int(r)] for r in rows]
    assert all(out[i] is seq[int(r)] for i, r in enumerate(rows))


# pandas is optional: only the frame tests below skip without it.
@pytest.fixture
def ne2r_pd():
    return pytest.importorskip("pandas")


def _ne2r_frame(pd, n=120):
    return pd.DataFrame({
        "i8": np.arange(n, dtype=np.int8),
        "i32": np.arange(n, dtype=np.int32) * 3,
        "u16": np.arange(n, dtype=np.uint16),
        "f32": np.linspace(-1, 1, n, dtype=np.float32),
        "f64": np.linspace(0, 5, n),
        "b": (np.arange(n) % 3 == 0),
    })


def test_resample_frame_mixed_dtypes_equal_iloc(ne2r_pd):
    df = _ne2r_frame(ne2r_pd)
    marker = np.arange(len(df), dtype=np.int64)
    out, idx = rs.resample(df, marker, n_samples=90, random_state=4)
    rows = _ne2r_rows_of(idx)
    expect = df.iloc[rows].to_numpy()
    got = np.asarray(out)
    assert got.dtype == expect.dtype
    if expect.dtype.hasobject:
        # A frame mixing bool with numeric columns materializes as object
        # cells (pandas iloc().to_numpy() does too); object storage holds
        # references, so compare each cell's type and value, not bytes.
        assert got.shape == expect.shape
        for g, e in zip(got.ravel(), expect.ravel()):
            assert type(g) is type(e) and g == e
    else:
        assert np.array_equal(got.view(np.uint8), np.ascontiguousarray(expect).view(np.uint8))


def test_resample_tz_period_interval_cells(ne2r_pd):
    pd = ne2r_pd
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
        rows = _ne2r_rows_of(idx)
        expect = s.iloc[rows].to_numpy(dtype=object) if s.dtype == object or str(s.dtype).startswith(("period", "interval", "datetime64[ns,")) else s.iloc[rows].to_numpy()
        got = np.asarray(out)
        assert len(got) == n
        for g, e in zip(got, expect):
            assert (pd.isna(g) and pd.isna(e)) or g == e


@pytest.mark.parametrize("dtype", ["datetime64[s]", "timedelta64[ns]"])
def test_resample_datetime_and_timedelta_arrays_without_pandas(dtype):
    # NumPy exports no PEP3118 buffer for 'M'/'m'; the rows operation reads
    # the output's address from its array interface (w3-integrate-11).
    n = 12
    x = np.arange(n).astype(dtype)
    marker = np.arange(n, dtype=np.int64)
    out, idx = rs.resample(x, marker, n_samples=n, random_state=8, replace=False)
    rows = _ne2r_rows_of(idx)
    assert np.asarray(out).dtype == x.dtype
    assert np.asarray(out).tobytes() == x[rows].tobytes()
