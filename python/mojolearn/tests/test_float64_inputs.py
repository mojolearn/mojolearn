"""float64 inputs of the float32-computing estimators (Andrew, 2026-11-27).

Every high-level estimator that computes in float32 accepts a float64 X (and
float64 targets and sample weights where float32 is what it computes in). The
native boundary converts it ONCE with one rule on every device and side:
IEEE round-to-nearest-even, NaN stays NaN, an infinity stays that infinity
(the estimator then treats it as it treats a float32 infinity), and a finite
value outside float32's range is refused with a ValueError naming its column.
A float32 C-order input is borrowed as before (no extra copy; no bit moves).

Per family the float64 result is, bit for bit, the result of the same values
pre-cast with ``numpy.astype(np.float32)``. The rule itself is
core/admitted_input.mojo (`admitted_cast_f32`, `admitted_f64_overflow_refuse`)
and core/admitted_input_nd.mojo (`admitted_cast_nd_f32`); its native test is
checks/native_tests/admitted_f64_narrow.mojo.

Kept strict, and tested where they live: the linalg and ops surface
(test_linalg_boundary.py, test_linalg_identity.py), saved-model loading
(test_host_model_kmeans.py), the training primitives, optimizers and neural
layers (test_neural_public_coverage_p1.py, test_neural_coverage.py).
"""
import struct

import numpy as np
import pytest

import mojolearn as ml

# float32's largest finite value, and the float64 values just past it: the
# first rounds back to it (kept), the second rounds to infinity (refused).
_F32_MAX = float(np.finfo(np.float32).max)
_KEPT = struct.unpack("<d", struct.pack("<Q", 0x47EFFFFFEFFFFFFF))[0]   # below max + half ulp
_REFUSED = struct.unpack("<d", struct.pack("<Q", 0x47EFFFFFF0000000))[0]  # max + half ulp: ties to even = inf
# A tie between two float32 neighbors: round-to-nearest-even and a
# round-half-away rule disagree here.
_TIE = struct.unpack("<d", struct.pack("<Q", 0x3FF0000010000000))[0]


def _bits(a):
    return np.ascontiguousarray(np.asarray(a, dtype=np.float32)).view(np.uint32)


def _same(a, b):
    return np.array_equal(_bits(a), _bits(b))


def _wide(rows, cols, seed):
    """float64 values that are not float32 values, with the tie case in."""
    x = np.random.default_rng(seed).standard_normal((rows, cols))
    x[0, 0] = _TIE
    return x


def _pair(rows, cols, seed):
    x64 = _wide(rows, cols, seed)
    return x64, x64.astype(np.float32)


def test_numpy_cast_is_the_rule():
    # The reference the tests compare against: numpy's astype is IEEE
    # round-to-nearest-even, keeps a value that rounds to float32's maximum and
    # makes infinity of one that rounds past it (which the product refuses).
    assert np.float32(_KEPT) == np.float32(_F32_MAX)
    assert np.isinf(np.array([_REFUSED]).astype(np.float32))[0]
    assert np.array([_TIE]).astype(np.float32).view(np.uint32)[0] == 0x3F800000


# ---------------------------------------------------------------- the shared X admission

@pytest.mark.parametrize("make", [
    pytest.param(lambda: ml.LinearRegression(), id="LinearRegression"),
    pytest.param(lambda: ml.Ridge(alpha=1.0), id="Ridge"),
])
def test_linear_float64_x_is_the_float32_bits(make):
    x64, x32 = _pair(64, 5, 1)
    # y and the weights are float32 values held in float64: an estimator
    # that keeps a float64 target computes on the same values either way, so
    # the comparison isolates the X conversion.
    y64 = np.random.default_rng(2).standard_normal(64).astype(np.float32).astype(np.float64)
    w64 = (np.random.default_rng(3).random(64) + 0.5).astype(np.float32).astype(np.float64)
    wide = make().fit(x64, y64, sample_weight=w64)
    narrow = make().fit(x32, y64.astype(np.float32), sample_weight=w64.astype(np.float32))
    assert _same(wide.coef_, narrow.coef_) and _same(wide.intercept_, narrow.intercept_)
    assert _same(wide.predict(x64), narrow.predict(x32))


def test_kmeans_float64_is_the_float32_bits():
    x64, x32 = _pair(200, 4, 4)
    wide = ml.KMeans(n_clusters=3, n_init=1, random_state=0).fit(x64)
    narrow = ml.KMeans(n_clusters=3, n_init=1, random_state=0).fit(x32)
    assert _same(wide.cluster_centers_, narrow.cluster_centers_)
    assert np.array_equal(np.asarray(wide.predict(x64)), np.asarray(narrow.predict(x32)))
    assert _same(wide.transform(x64), narrow.transform(x32))


def test_neighbors_float64_is_the_float32_bits():
    x64, x32 = _pair(120, 3, 5)
    y = np.random.default_rng(6).standard_normal(120).astype(np.float32)
    wide = ml.KNeighborsRegressor(n_neighbors=4).fit(x64, y)
    narrow = ml.KNeighborsRegressor(n_neighbors=4).fit(x32, y)
    assert _same(wide.predict(x64), narrow.predict(x32))


def test_pca_float64_is_the_float32_bits():
    x64, x32 = _pair(90, 6, 7)
    wide = ml.PCA(n_components=3).fit(x64)
    narrow = ml.PCA(n_components=3).fit(x32)
    assert _same(wide.components_, narrow.components_)
    assert _same(wide.transform(x64), narrow.transform(x32))


def test_tree_float64_is_the_float32_bits():
    x64, x32 = _pair(150, 4, 8)
    y = (x32[:, 0] > 0).astype(np.int64)
    wide = ml.DecisionTreeClassifier(max_depth=4, random_state=0).fit(x64, y)
    narrow = ml.DecisionTreeClassifier(max_depth=4, random_state=0).fit(x32, y)
    assert _same(wide.predict_proba(x64), narrow.predict_proba(x32))


def test_hdbscan_float64_query_is_the_float32_bits():
    x32 = _wide(300, 2, 9).astype(np.float32)
    model = ml.HDBSCAN(min_cluster_size=15, prediction_data=True).fit(x32)
    q64 = _wide(17, 2, 10)
    l64, p64 = ml.hdbscan.approximate_predict(model, q64)
    l32, p32 = ml.hdbscan.approximate_predict(model, q64.astype(np.float32))
    assert np.array_equal(np.asarray(l64), np.asarray(l32)) and _same(p64, p32)


def test_float32_c_order_input_is_borrowed_unchanged():
    # No bit moves for a float32 input: the float64 path changed nothing for it.
    x32 = _wide(64, 5, 11).astype(np.float32)
    before = x32.copy()
    ml.LinearRegression().fit(x32, x32[:, 0])
    assert x32.tobytes() == before.tobytes()


# ---------------------------------------------------------------- scalers

@pytest.mark.parametrize("scaler", [ml.StandardScaler, ml.MinMaxScaler])
def test_scaler_float64_is_the_float32_bits(scaler):
    x64, x32 = _pair(80, 4, 12)
    wide = scaler().fit(x64)
    narrow = scaler().fit(x32)
    assert _same(wide.transform(x64), narrow.transform(x32))
    assert _same(wide.inverse_transform(x64), narrow.inverse_transform(x32))


def test_scaler_float64_nan_stays_nan():
    # StandardScaler treats NaN as missing: a float64 NaN is that NaN.
    x64, x32 = _pair(40, 3, 13)
    x64[5, 1] = np.nan
    x32 = x64.astype(np.float32)
    wide = ml.StandardScaler().fit(x64)
    narrow = ml.StandardScaler().fit(x32)
    out = np.asarray(wide.transform(x64))
    assert np.isnan(out[5, 1]) and _same(out, narrow.transform(x32))


def test_scaler_float64_infinity_is_treated_as_float32_infinity():
    x = np.ones((4, 2))
    x[1, 0] = np.inf
    with pytest.raises(ValueError) as wide:
        ml.StandardScaler().fit(x)
    with pytest.raises(ValueError) as narrow:
        ml.StandardScaler().fit(x.astype(np.float32))
    assert str(wide.value) == str(narrow.value)


# ---------------------------------------------------------------- sequence and time series

def _series(n, seed, batch=None):
    rng = np.random.default_rng(seed)
    shape = (n,) if batch is None else (batch, n)
    return 10.0 + np.abs(rng.standard_normal(shape)).cumsum(axis=-1) * 0.1


def test_theta_float64_is_the_float32_bits():
    y64 = _series(60, 14, batch=3)
    wide = ml.Theta(season_length=1).fit(y64).predict(5)["mean"]
    narrow = ml.Theta(season_length=1).fit(y64.astype(np.float32)).predict(5)["mean"]
    assert _same(wide, narrow)


def test_croston_float64_is_the_float32_bits():
    y64 = np.where(np.random.default_rng(15).random(50) < 0.4, _series(50, 16), 0.0)
    wide = ml.CrostonClassic().fit(y64).predict(4)["mean"]
    narrow = ml.CrostonClassic().fit(y64.astype(np.float32)).predict(4)["mean"]
    assert _same(wide, narrow)


def test_ets_float64_is_the_float32_bits():
    y64 = _series(48, 17)
    wide = ml.ETS(season_length=1, model="AAN").fit(y64).predict(6)["mean"]
    narrow = ml.ETS(season_length=1, model="AAN").fit(y64.astype(np.float32)).predict(6)["mean"]
    assert _same(wide, narrow)


def test_garch_float64_is_the_float32_bits():
    r64 = np.random.default_rng(18).standard_normal(300)
    wide = ml.GARCH(p=1, q=1).fit(r64).forecast(3)
    narrow = ml.GARCH(p=1, q=1).fit(r64.astype(np.float32)).forecast(3)
    assert _same(np.asarray(wide), np.asarray(narrow))


def test_var_float64_is_the_float32_bits():
    y64 = np.random.default_rng(19).standard_normal((120, 3)).cumsum(axis=0)
    wide = ml.VAR(y64).fit(maxlags=2)
    narrow = ml.VAR(y64.astype(np.float32)).fit(maxlags=2)
    assert _same(wide.forecast(y64, steps=4), narrow.forecast(y64.astype(np.float32), steps=4))


def test_stl_float64_is_the_float32_bits():
    t = np.arange(96, dtype=np.float64)
    y64 = 0.05 * t + np.sin(2 * np.pi * t / 12) + 0.001 * np.random.default_rng(20).standard_normal(96)
    wide = ml.STL(y64, period=12).fit()
    narrow = ml.STL(y64.astype(np.float32), period=12).fit()
    assert _same(wide.seasonal, narrow.seasonal) and _same(wide.trend, narrow.trend)


def test_autoarima_float64_is_the_float32_bits():
    y64 = np.random.default_rng(21).standard_normal((2, 80)).cumsum(axis=1)
    wide = ml.AutoARIMA(y64).search(d=range(2), p=range(2), q=range(2)).fit()
    narrow = ml.AutoARIMA(y64.astype(np.float32)).search(d=range(2), p=range(2), q=range(2)).fit()
    assert _same(wide.forecast(3), narrow.forecast(3))


def test_prophet_float64_y_is_the_float32_bits():
    t = np.arange(120, dtype=np.float64)
    y64 = 10 + 0.01 * t + np.sin(2 * np.pi * t / 7) + 0.001 * np.random.default_rng(22).standard_normal(120)
    future = np.arange(120, 130, dtype=np.float64)
    wide = ml.ProphetForecaster().fit(t, y64).predict(future)
    narrow = ml.ProphetForecaster().fit(t, y64.astype(np.float32)).predict(future)
    assert _same(np.asarray(wide), np.asarray(narrow))


def test_lstm_float64_x_and_y_are_the_float32_bits():
    rng = np.random.default_rng(23)
    x64 = rng.standard_normal((24, 5, 3))
    y64 = rng.standard_normal(24)
    kw = dict(hidden_size=4, max_epochs=2, batch_size=8, random_state=0)
    wide = ml.LSTMRegressor(**kw).fit(x64, y64)
    narrow = ml.LSTMRegressor(**kw).fit(x64.astype(np.float32), y64.astype(np.float32))
    assert _same(wide.predict(x64), narrow.predict(x64.astype(np.float32)))


# ---------------------------------------------------------------- out of range: refused by column

def _overflow(rows=8, cols=4, column=2, value=_REFUSED):
    x = _wide(rows, cols, 24)
    x[3, column] = value
    return x


@pytest.mark.parametrize("call", [
    pytest.param(lambda x: ml.LinearRegression().fit(x, np.ones(len(x))), id="admit_x"),
    pytest.param(lambda x: ml.KMeans(n_clusters=2, n_init=1, random_state=0).fit(x), id="family"),
    pytest.param(lambda x: ml.StandardScaler().fit(x), id="scaler"),
    pytest.param(lambda x: ml.MinMaxScaler().fit(x), id="minmax"),
])
def test_out_of_range_float64_is_refused_naming_its_column(call):
    with pytest.raises(ValueError, match="column 2 holds a float64 value whose magnitude exceeds"):
        call(_overflow())
    with pytest.raises(ValueError, match="column 2 holds a float64 value"):
        call(_overflow(value=-_REFUSED))


def test_out_of_range_float64_names_the_smallest_column():
    x = _overflow(rows=40, cols=6, column=4)
    x[30, 1] = 1e300
    with pytest.raises(ValueError, match="column 1 holds a float64 value"):
        ml.LinearRegression().fit(x, np.ones(len(x)))


def test_out_of_range_float64_series_is_refused():
    # A batch of series (batch, n_obs): the column is the observation index;
    # a single 1-D series is one column (column 0).
    y = _series(40, 25, batch=3)
    y[1, 7] = 1e39
    with pytest.raises(ValueError, match="column 7 holds a float64 value"):
        ml.Theta(season_length=1).fit(y)
    with pytest.raises(ValueError, match="column 0 holds a float64 value"):
        ml.Theta(season_length=1).fit(y[1])


def test_out_of_range_float64_sequence_x_is_refused():
    x = np.zeros((4, 3, 2))
    x[1, 2, 1] = -1e39
    with pytest.raises(ValueError, match="column 1 holds a float64 value"):
        ml.LSTMRegressor(hidden_size=2, max_epochs=1).fit(x, np.zeros(4))


def test_out_of_range_float64_pandas_column_is_refused():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [1.0, _REFUSED, 3.0, 4.0]})
    with pytest.raises(ValueError, match="column 1 holds a float64 value"):
        ml.LinearRegression().fit(frame, np.ones(4))


def test_value_that_rounds_to_float32_max_is_kept():
    x64 = _wide(32, 3, 26)
    x64[4, 1] = _KEPT
    wide = ml.StandardScaler().fit(x64)
    narrow = ml.StandardScaler().fit(x64.astype(np.float32))
    assert _same(wide.mean_, narrow.mean_)


# ---------------------------------------------------------------- strict entries stay strict

def test_linalg_still_refuses_float64():
    with pytest.raises(TypeError):
        ml.linalg.qr(np.eye(3))
