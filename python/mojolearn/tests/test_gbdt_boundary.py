# SPDX-License-Identifier: Apache-2.0
"""GradientBoosting public boundary: returned objects, types and dtypes, refusals by name, input forms
(lists, object, unicode and bytes arrays, pandas and Arrow columns), the saved-model readers' admission and the
scikit-learn style adapters. The numerical digests of these estimators are the verify lanes (tools/identity_break.py
gbdt-*, tools/identity_lanes/trees.py)."""
import pickle

import numpy as np
import pytest

import mojolearn
import mojolearn as ml
from mojolearn import ExperimentalTwoLevelFeatureFreq
from mojolearn._gbdt_host import parse_model_text
from mojolearn.exceptions import NotFittedError
# the hand-written CTR model texts stay beside the host CTR lookups that also use them
from mojolearn.tests.test_gbdt_host_ctr import (
    CTR_TEXT as _HCTR_CTR_TEXT, TENSOR_TEXT as _HCTR_TENSOR_TEXT, _raises as _hctr_raises)

try:
    import pandas as _gci_pd
except ImportError:  # the pandas and Arrow input forms below skip without it
    _gci_pd = None
_gci_needs_pandas = pytest.mark.skipif(_gci_pd is None, reason="pandas is not installed")


# ---------------------------------------------------------------- moved from
# test_feature_freq_experimental_surface.py (vdv-pytest-a)


def test_experimental_feature_freq_validates_and_forwards():
    X = np.asarray(
        [[0, 0, -2.5], [0, 1, 0.25], [1, 0, 3.5], [1, 1, 8.0]],
        dtype=np.float32,
    )
    y = np.asarray([-5, -3, 3, 5], dtype=np.float32)

    def fit(**kw):
        est = ExperimentalTwoLevelFeatureFreq(sources=[0, 1], l2_leaf_reg=4.0, random_state=7, **kw)
        assert est.fit(X, y) is est
        return est

    est = fit(learning_rate=0.2)
    assert est.model_
    assert est.n_features_in_ == 3
    # the options reach the fit
    assert fit(learning_rate=0.5).model_ != est.model_


@pytest.mark.parametrize(
    "X,sources,needle",
    [
        ([[0, 0], [0, 1], [2, 0], [2, 1]], [0, 1], "densely coded"),
        ([[0, 0], [0, 1], [1, 0], [1, 1]], [0, 2], "outside"),
        ([[0, 0], [0, 1], [1.5, 0], [1, 1]], [0, 1], "integer"),
    ],
)
def test_experimental_feature_freq_refusals(X, sources, needle):
    est = ExperimentalTwoLevelFeatureFreq(sources=sources)
    with pytest.raises(ValueError, match=needle):
        est.fit(np.asarray(X, dtype=np.float32), np.zeros(4, np.float32))


@pytest.mark.parametrize(
    "weights,needle",
    [
        ([1.0, 1.0], "one value per row"),
        ([1.0, -1.0, 1.0, 1.0], "finite and non-negative"),
        ([0.0, 0.0, 0.0, 0.0], "positive sum"),
        ([1.0, float("nan"), 1.0, 1.0], "finite and non-negative"),
        ([1.0, float("inf"), 1.0, 1.0], "finite and non-negative"),
    ],
)
def test_experimental_feature_freq_weight_refusals(weights, needle):
    X = np.asarray(
        [[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32
    )
    est = ExperimentalTwoLevelFeatureFreq(sources=[0, 1])
    with pytest.raises(ValueError, match=needle):
        est.fit(X, np.zeros(4, np.float32), sample_weight=weights)


def test_experimental_feature_freq_refuses_constant_numeric_column():
    X = np.asarray(
        [[0, 0, 7], [0, 1, 7], [1, 0, 7], [1, 1, 7]], dtype=np.float32
    )
    est = ExperimentalTwoLevelFeatureFreq(sources=[0, 1])
    with pytest.raises(ValueError, match="numeric columns must vary; column 2"):
        est.fit(X, np.zeros(4, np.float32))


def test_experimental_feature_freq_forwards_weights():
    # both columns train sample_weight here since lane gbdt-cpu-rest
    # (2026-11-23): the host fit reads them as the device's has_w arm
    X = np.asarray(
        [[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32
    )
    y = np.asarray([-2, -1, 1, 2], dtype=np.float32)

    def model(weights):
        return ExperimentalTwoLevelFeatureFreq(sources=[0, 1]).fit(X, y, sample_weight=weights).model_

    unit = ExperimentalTwoLevelFeatureFreq(sources=[0, 1]).fit(X, y).model_
    assert model(np.asarray([1, 2, 3, 4], dtype=np.float32)) != unit


# ---------------------------------------------------------------- moved from
# test_gbdt_feature_importances.py (vdv-pytest-a)
# `feature_importances_` (lane mf-imp6): the returned ndarray, a fresh array per read, the not-fitted error in
# scikit-learn's words, the adapters, save/load, pickle and HostGBDT, set_params and scikit-learn's SelectFromModel.


def _imp_data(seed=11, n=400, d=5):
    rng = np.random.default_rng(seed)
    X = rng.random((n, d), dtype=np.float32)
    y = (4.0 * X[:, 2] + 0.5 * X[:, 0] + 0.01 * rng.normal(size=n)).astype(np.float32)
    return X, y


def _imp_check(imp, d):
    assert type(imp) is np.ndarray
    assert imp.shape == (d,)
    assert imp.dtype == np.float64
    assert np.all(imp >= 0.0)
    assert abs(float(imp.sum()) - 1.0) <= 1e-6


def test_each_read_is_a_fresh_array():
    X, y = _imp_data()
    model = ml.GradientBoosting(n_estimators=8).fit(X, y)
    first = model.feature_importances_
    kept = first.copy()
    first[:] = -1.0
    np.testing.assert_array_equal(model.feature_importances_, kept)


@pytest.mark.parametrize("make", [
    lambda: ml.GradientBoosting(n_estimators=4),
    lambda: ml.GradientBoostingRegressor(n_estimators=4),
    lambda: ml.GradientBoostingClassifier(n_estimators=4),
])
def test_not_fitted_is_sklearns_error(make):
    est = make()
    with pytest.raises(NotFittedError, match="instance is not fitted yet. Call 'fit' with appropriate arguments"):
        est.feature_importances_


def test_adapters_expose_the_learners_importances():
    X, y = _imp_data()
    reg = ml.GradientBoostingRegressor(n_estimators=10).fit(X, y)
    own = ml.GradientBoosting(n_estimators=10).fit(X, y)
    _imp_check(reg.feature_importances_, X.shape[1])
    np.testing.assert_array_equal(reg.feature_importances_, own.feature_importances_)
    codes = (y > np.median(y)).astype(np.int64)
    clf = ml.GradientBoostingClassifier(n_estimators=10).fit(X, codes)
    _imp_check(clf.feature_importances_, X.shape[1])


def test_multiclass_sums_its_dimensions():
    X, y = _imp_data()
    codes = np.digitize(y, np.quantile(y, [0.33, 0.66])).astype(np.float32)
    model = ml.GradientBoosting(loss="MultiClass", n_estimators=10).fit(X, codes)
    _imp_check(model.feature_importances_, X.shape[1])


def test_save_load_pickle_and_host_twin(tmp_path):
    X, y = _imp_data()
    model = ml.GradientBoosting(n_estimators=8).fit(X, y)
    path = tmp_path / "gb.npz"
    model.save(path)
    np.testing.assert_array_equal(ml.GradientBoosting.load(path).feature_importances_, model.feature_importances_)
    np.testing.assert_array_equal(ml.HostGBDT.from_file(path).feature_importances_, model.feature_importances_)
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(model)).feature_importances_, model.feature_importances_)


def test_set_params_clears_them():
    X, y = _imp_data()
    model = ml.GradientBoosting(n_estimators=4).fit(X, y)
    model.set_params(n_estimators=5)
    with pytest.raises(NotFittedError):
        model.feature_importances_


def test_sklearn_select_from_model_reads_them():
    feature_selection = pytest.importorskip("sklearn.feature_selection")
    X, y = _imp_data()
    sel = feature_selection.SelectFromModel(ml.GradientBoostingRegressor(n_estimators=10), max_features=1,
                                            threshold=-np.inf).fit(X, y)
    assert list(np.flatnonzero(sel.get_support())) == [2]


# ---------------------------------------------------------------- moved from test_gbdt_fit_loop.py (vdv-pytest-a)
# `fit_loop_` (lane ab-loop-mode): a tuple (int, float) after fit, AttributeError before fit and on a loaded model,
# and the adapters copy it.


def _fl_data(seed=5, n=300, d=4):
    rng = np.random.default_rng(seed)
    X = rng.random((n, d), dtype=np.float32)
    y = (2.0 * X[:, 0] + X[:, 1]).astype(np.float32)
    return X, y


def _fl_check(loop, rounds):
    assert type(loop) is tuple and len(loop) == 2
    assert type(loop[0]) is int and type(loop[1]) is float
    assert loop[0] == rounds
    assert loop[1] >= 0.0


@pytest.mark.parametrize("grow_policy", ["SymmetricTree", "Depthwise"])
def test_fit_loop_after_fit(grow_policy):
    X, y = _fl_data()
    model = ml.GradientBoosting(n_estimators=6, max_depth=3, grow_policy=grow_policy).fit(X, y)
    _fl_check(model.fit_loop_, 6)


def test_fit_loop_raises_before_fit():
    with pytest.raises(AttributeError):
        ml.GradientBoosting(n_estimators=4).fit_loop_


def test_fit_loop_not_carried_by_the_archive(tmp_path):
    X, y = _fl_data()
    model = ml.GradientBoosting(n_estimators=4).fit(X, y)
    path = tmp_path / "model.npz"
    model.save(str(path))
    loaded = ml.GradientBoosting.load(str(path))
    with pytest.raises(AttributeError):
        loaded.fit_loop_


def test_adapters_copy_fit_loop():
    X, y = _fl_data()
    _fl_check(ml.GradientBoostingRegressor(n_estimators=5).fit(X, y).fit_loop_, 5)
    labels = (y > np.median(y)).astype(np.int64)
    _fl_check(ml.GradientBoostingClassifier(n_estimators=5).fit(X, labels).fit_loop_, 5)


# ---------------------------------------------------------------- moved from test_gbdt_host_ctr.py (vdv-pytest-a)
# The saved CTR records parse with the Mojo reader's checks, and what the GPU predict refuses is refused at load by
# name.


def test_ctr_records_parse_with_the_column_plan():
    a = parse_model_text(_HCTR_CTR_TEXT)
    assert a["n_features"] == 1 and a["n_input_features"] == 1
    assert a["n_ctr_tables"] == 1 and a["n_tensor_tables"] == 0 and a["n_ctr_counts"] == 3
    t = parse_model_text(_HCTR_TENSOR_TEXT)
    assert t["n_features"] == 3 and t["n_input_features"] == 2
    assert t["n_tensor_tables"] == 1 and t["n_ctr_counts"] == 4


def test_unsupported_ctr_type_refuses_by_name():
    for name in ("Buckets", "BinarizedTargetMeanValue", "FloatTargetMeanValue"):
        _hctr_raises(lambda: parse_model_text(_HCTR_CTR_TEXT.replace("type FeatureFreq", f"type {name}")), name)
    _hctr_raises(lambda: parse_model_text(_HCTR_CTR_TEXT.replace("type FeatureFreq", "type Mystery")), "Mystery")


def test_what_predict_floats_refuses_is_refused_at_load():
    _hctr_raises(lambda: parse_model_text(_HCTR_CTR_TEXT.replace("ctr_columns 1", "ctr_columns 2")),
                 "2 CTR columns and 1 CTR tables")
    _hctr_raises(lambda: parse_model_text(_HCTR_CTR_TEXT.replace("ctr_entry 0 2 1\n", "")), "declares 3 entries")
    _hctr_raises(lambda: parse_model_text(_HCTR_CTR_TEXT.replace("type ctr", "type float")), "says type 'float'")
    _hctr_raises(lambda: parse_model_text(_HCTR_TENSOR_TEXT.replace("counts 4 5 3 1 1", "counts 4 5 3 1 2")),
                 "do not sum to denominator")
    _hctr_raises(lambda: parse_model_text(_HCTR_TENSOR_TEXT.replace("sources 2 0 1", "sources 2 1 0")),
                 "canonical")


# ---------------------------------------------------------------- moved from
# tests/api/test_gbdt_category_inputs.py (vdv-pytest-a)
# The input forms `GradientBoosting(cat_features=...)` admits for its categorical columns
# (bindings/gbdt_category_unpack.mojo), at fit and at predict, and the refusals by name: every accepted form of the
# same data gives the same predictions.

_GCI_N = 240
_GCI_CATS = ["red", "green", "blue", "teal"]


def _gci_data():
    rng = np.random.default_rng(3)
    cat = [_GCI_CATS[i % 4] for i in range(_GCI_N)]
    num = rng.normal(size=_GCI_N).astype(np.float64)
    y = np.array([float(i % 4) + 0.1 * num[i] for i in range(_GCI_N)], dtype=np.float32)
    return cat, num, y


def _gci_model():
    return mojolearn.GradientBoosting(n_estimators=8, max_depth=3, cat_features=[0], device="cpu")


def _gci_fit_predict(X, y, X_test=None):
    model = _gci_model().fit(X, y)
    return model.predict(X if X_test is None else X_test)


@pytest.fixture(scope="module")
def gci_reference():
    cat, num, y = _gci_data()
    rows = [[cat[i], float(num[i])] for i in range(_GCI_N)]
    return _gci_fit_predict(rows, y)


def test_object_array_matches_list_rows(gci_reference):
    cat, num, y = _gci_data()
    X = np.empty((_GCI_N, 2), dtype=object)
    X[:, 0] = cat
    X[:, 1] = num
    np.testing.assert_array_equal(_gci_fit_predict(X, y), gci_reference)
    # a strided object view is read through its own strides
    wide = np.empty((_GCI_N, 4), dtype=object)
    wide[:, 0] = cat
    wide[:, 2] = num
    np.testing.assert_array_equal(_gci_fit_predict(wide[:, ::2], y), gci_reference)


def test_unicode_and_bytes_arrays_with_numeric_text(gci_reference):
    cat, num, y = _gci_data()
    # repr round-trips a float64, so the parsed column holds the same values
    U = np.array([[cat[i], repr(float(num[i]))] for i in range(_GCI_N)])
    assert U.dtype.kind == "U"
    np.testing.assert_array_equal(_gci_fit_predict(U, y), gci_reference)
    S = U.astype("S")
    np.testing.assert_array_equal(_gci_fit_predict(S, y), gci_reference)


def test_unicode_array_numeric_column_refused_in_words():
    cat, num, y = _gci_data()
    U = np.array([[cat[i], "x" if i == 7 else "1.0"] for i in range(_GCI_N)])
    with pytest.raises(ValueError, match=r"X\[7\]\[1\] is not a number"):
        _gci_model().fit(U, y)


def _gci_frame(cat, num, dtype):
    df = _gci_pd.DataFrame({"c": _gci_pd.Series(cat, dtype=dtype), "x": num})
    return df


@_gci_needs_pandas
@pytest.mark.parametrize("dtype", [object, "category"])
def test_pandas_columns_match_list_rows(gci_reference, dtype):
    cat, num, y = _gci_data()
    np.testing.assert_array_equal(_gci_fit_predict(_gci_frame(cat, num, dtype), y), gci_reference)


@_gci_needs_pandas
def test_pandas_python_string_column(gci_reference):
    cat, num, y = _gci_data()
    try:
        dtype = _gci_pd.StringDtype("python")
    except TypeError:
        pytest.skip("this pandas has no Python-backed string dtype")
    np.testing.assert_array_equal(_gci_fit_predict(_gci_frame(cat, num, dtype), y), gci_reference)


@_gci_needs_pandas
def test_pandas_category_codes_are_not_the_categories(gci_reference):
    # the same values with categories in another order: other codes, same categories
    cat, num, y = _gci_data()
    df = _gci_frame(cat, num, _gci_pd.CategoricalDtype(categories=list(reversed(_GCI_CATS))))
    np.testing.assert_array_equal(_gci_fit_predict(df, y), gci_reference)


@_gci_needs_pandas
def test_pandas_missing_is_one_category():
    cat, num, y = _gci_data()
    with_missing = [None if i % 9 == 0 else cat[i] for i in range(_GCI_N)]
    rows = [[with_missing[i], float(num[i])] for i in range(_GCI_N)]
    expected = _gci_fit_predict(rows, y)
    as_category = _gci_frame(with_missing, num, "category")  # None -> code -1
    np.testing.assert_array_equal(_gci_fit_predict(as_category, y), expected)
    as_na = _gci_frame([_gci_pd.NA if v is None else v for v in with_missing], num, object)
    np.testing.assert_array_equal(_gci_fit_predict(as_na, y), expected)


@_gci_needs_pandas
def test_predict_maps_unseen_values_through_the_stored_map():
    cat, num, y = _gci_data()
    model = _gci_model().fit(_gci_frame(cat, num, "category"), y)
    test = _gci_frame(["red", "violet", "blue"], num[:3], "category")
    got = model.predict(test)
    rows = model.predict([["red", float(num[0])], ["violet", float(num[1])], ["blue", float(num[2])]])
    np.testing.assert_array_equal(got, rows)


@_gci_needs_pandas
def test_numeric_category_column_and_numeric_feature():
    cat, num, y = _gci_data()
    codes = [i % 4 for i in range(_GCI_N)]
    plain = _gci_pd.DataFrame({"c": codes, "x": num, "z": np.arange(_GCI_N, dtype=np.int64) % 5})
    as_category = plain.astype({"c": "category", "z": "category"})  # z is a numeric feature
    np.testing.assert_array_equal(_gci_fit_predict(as_category, y), _gci_fit_predict(plain, y))


def _gci_arrow_dtype(storage):
    pytest.importorskip("pyarrow")
    try:
        return _gci_pd.StringDtype(storage)
    except (TypeError, ImportError):
        pytest.skip("this pandas has no " + storage + " string dtype")


@_gci_needs_pandas
def test_arrow_string_column_matches_list_rows(gci_reference):
    # pandas 2 ``string[pyarrow]``; pandas 3's default ``str`` with pyarrow
    cat, num, y = _gci_data()
    df = _gci_frame(cat, num, _gci_arrow_dtype("pyarrow"))
    np.testing.assert_array_equal(_gci_fit_predict(df, y), gci_reference)


@_gci_needs_pandas
def test_arrow_large_string_and_chunked(gci_reference):
    pa = pytest.importorskip("pyarrow")
    cat, num, y = _gci_data()
    half = _GCI_N // 2
    for typ in (pa.string(), pa.large_string()):
        chunked = pa.chunked_array([pa.array(cat[:half], type=typ), pa.array(cat[half:], type=typ)])
        column = _gci_pd.Series(_gci_pd.arrays.ArrowExtensionArray(chunked))
        df = _gci_pd.DataFrame({"c": column, "x": num})
        np.testing.assert_array_equal(_gci_fit_predict(df, y), gci_reference)


@_gci_needs_pandas
def test_arrow_sliced_chunk_offsets(gci_reference):
    # a chunk that is a slice of a larger array: its offset is not 0
    pa = pytest.importorskip("pyarrow")
    cat, num, y = _gci_data()
    padded = pa.array(["pad"] * 7 + cat + ["pad"] * 3).slice(7, _GCI_N)
    column = _gci_pd.Series(_gci_pd.arrays.ArrowExtensionArray(pa.chunked_array([padded])))
    np.testing.assert_array_equal(_gci_fit_predict(_gci_pd.DataFrame({"c": column, "x": num}), y), gci_reference)


@_gci_needs_pandas
def test_arrow_dictionary_column(gci_reference):
    pa = pytest.importorskip("pyarrow")
    cat, num, y = _gci_data()
    encoded = pa.array(cat).dictionary_encode()
    column = _gci_pd.Series(_gci_pd.arrays.ArrowExtensionArray(pa.chunked_array([encoded])))
    np.testing.assert_array_equal(_gci_fit_predict(_gci_pd.DataFrame({"c": column, "x": num}), y), gci_reference)


@_gci_needs_pandas
def test_arrow_missing_is_one_category():
    cat, num, y = _gci_data()
    with_missing = [None if i % 9 == 0 else cat[i] for i in range(_GCI_N)]
    rows = [[with_missing[i], float(num[i])] for i in range(_GCI_N)]
    expected = _gci_fit_predict(rows, y)
    df = _gci_frame(with_missing, num, _gci_arrow_dtype("pyarrow"))
    np.testing.assert_array_equal(_gci_fit_predict(df, y), expected)


@_gci_needs_pandas
def test_arrow_categories_of_a_categorical(gci_reference):
    # pandas 3 keeps str categories in an Arrow-backed Index
    cat, num, y = _gci_data()
    dtype = _gci_pd.CategoricalDtype(_gci_pd.Index(_GCI_CATS, dtype=_gci_arrow_dtype("pyarrow")))
    np.testing.assert_array_equal(_gci_fit_predict(_gci_frame(cat, num, dtype), y), gci_reference)


@_gci_needs_pandas
def test_arrow_fit_object_predict_agree(gci_reference):
    # the same text is the same category whatever storage predict receives
    cat, num, y = _gci_data()
    model = _gci_model().fit(_gci_frame(cat, num, _gci_arrow_dtype("pyarrow")), y)
    np.testing.assert_array_equal(model.predict(_gci_frame(cat, num, object)), gci_reference)


@_gci_needs_pandas
def test_arrow_column_outside_cat_features_refused():
    cat, num, y = _gci_data()
    df = _gci_pd.DataFrame({"c": _gci_pd.Series(cat, dtype=object),
                            "t": _gci_pd.Series(cat, dtype=_gci_arrow_dtype("pyarrow")), "x": num})
    with pytest.raises(TypeError, match="outside cat_features"):
        _gci_model().fit(df, y)


@_gci_needs_pandas
def test_arrow_numeric_type_refused_by_name():
    pa = pytest.importorskip("pyarrow")
    cat, num, y = _gci_data()
    column = _gci_pd.Series(_gci_pd.arrays.ArrowExtensionArray(pa.chunked_array([pa.array(list(range(_GCI_N)))])))
    with pytest.raises(TypeError, match="Int64Array"):
        _gci_model().fit(_gci_pd.DataFrame({"c": column, "x": num}), y)


@_gci_needs_pandas
def test_text_in_numeric_pandas_column_refused():
    cat, num, y = _gci_data()
    df = _gci_pd.DataFrame({"c": cat, "x": [str(v) for v in num]}, dtype=object)
    with pytest.raises(TypeError, match="only the cat_features columns may hold text"):
        _gci_model().fit(df, y)


# ---------------------------------------------------------------- moved from
# tests/api/test_host_gbdt_category_map.py (vdv-pytest-a)
# `HostGBDT` applies a saved model's category map with the same admission and apply functions
# (bindings/gbdt_category_unpack.mojo): the same file gives the same bits from either class on every accepted input
# form, and refuses in the same words.

_HGC_N = 240
_HGC_CATS = ["red", "green", "blue", "teal"]
_HGC_MANY = ["c%02d" % i for i in range(40)]


def _hgc_data(cats=_HGC_CATS):
    rng = np.random.default_rng(5)
    cat = [cats[(7 * i) % len(cats)] for i in range(_HGC_N)]
    cat2 = [cats[(3 * i + 1) % len(cats)] for i in range(_HGC_N)]
    num = rng.normal(size=_HGC_N).astype(np.float64)
    y = np.array([float(cats.index(cat[i]) % 4) + 0.1 * num[i] for i in range(_HGC_N)], dtype=np.float32)
    return cat, cat2, num, y


def _hgc_rows(cat, num):
    return [[cat[i], float(num[i])] for i in range(len(cat))]


@pytest.fixture(scope="module")
def hgc_saved(tmp_path_factory):
    cat, _, num, y = _hgc_data()
    model = mojolearn.GradientBoosting(n_estimators=8, max_depth=3, cat_features=[0], device="cpu")
    model.fit(_hgc_rows(cat, num), y)
    path = str(tmp_path_factory.mktemp("hostgbdt") / "cat.npz")
    model.save(path)
    return model, path


def _hgc_host(path):
    return mojolearn.HostGBDT.from_file(path)


def _hgc_queries():
    cat, _, num, _ = _hgc_data()
    test_cat = cat[:20] + ["violet", None, "red", 3]
    test_num = list(num[:24])
    return test_cat, test_num


def test_host_matches_model_on_list_rows_with_unseen_and_none(hgc_saved):
    model, path = hgc_saved
    cat, num = _hgc_queries()
    X = _hgc_rows(cat, num)
    np.testing.assert_array_equal(np.asarray(_hgc_host(path).predict(X)), np.asarray(model.predict(X)))


def test_host_matches_model_on_every_input_form(hgc_saved):
    model, path = hgc_saved
    host = _hgc_host(path)
    cat, _, num, _ = _hgc_data()
    cat, num = cat[:30], num[:30]
    expected = np.asarray(model.predict(_hgc_rows(cat, num)))
    obj = np.empty((30, 2), dtype=object)
    obj[:, 0] = cat
    obj[:, 1] = num
    np.testing.assert_array_equal(np.asarray(host.predict(obj)), expected)
    U = np.array([[cat[i], repr(float(num[i]))] for i in range(30)])
    np.testing.assert_array_equal(np.asarray(host.predict(U)), expected)
    np.testing.assert_array_equal(np.asarray(host.predict(U.astype("S"))), expected)
    pd = pytest.importorskip("pandas")
    for dtype in (object, "category"):
        df = pd.DataFrame({"c": pd.Series(cat, dtype=dtype), "x": num})
        np.testing.assert_array_equal(np.asarray(host.predict(df)), expected)


def test_host_refuses_in_the_models_words(hgc_saved):
    model, path = hgc_saved
    host = _hgc_host(path)
    cat, num = _hgc_queries()
    bad_inputs = (
        [[cat[0], float(num[0]), 1.0]],              # width
        [[cat[0], "x"]],                             # text in a numeric column
        [[object(), float(num[0])]],                 # a cell of no admitted kind
    )
    for X in bad_inputs:
        with pytest.raises(Exception) as from_model:
            model.predict(X)
        with pytest.raises(Exception) as from_host:
            host.predict(X)
        assert type(from_host.value) is type(from_model.value)
        assert str(from_host.value) == str(from_model.value)


def test_digest_equal_and_map_kept(hgc_saved, tmp_path):
    model, path = hgc_saved
    host = _hgc_host(path)
    assert host.digest_ == model.digest_
    np.testing.assert_array_equal(np.asarray(host.category_map_), np.asarray(model.category_map_))
    restored = pickle.loads(pickle.dumps(host))
    cat, num = _hgc_queries()
    X = _hgc_rows(cat, num)
    np.testing.assert_array_equal(np.asarray(restored.predict(X)), np.asarray(model.predict(X)))


def test_constructor_takes_the_map(hgc_saved):
    model, path = hgc_saved
    host = mojolearn.HostGBDT(loss=model.loss, text=model.model_, n_features_in=model.n_features_in_,
                              approx_dim=model.approx_dim_, n_classes=model.n_classes_, bias=model.bias_,
                              category_map=model.category_map_)
    cat, num = _hgc_queries()
    X = _hgc_rows(cat, num)
    np.testing.assert_array_equal(np.asarray(host.predict(X)), np.asarray(model.predict(X)))
    with pytest.raises(ValueError, match="refusing to cast a model file"):
        mojolearn.HostGBDT(loss=model.loss, text=model.model_, n_features_in=model.n_features_in_,
                           approx_dim=model.approx_dim_, n_classes=model.n_classes_, bias=model.bias_,
                           category_map=np.asarray(model.category_map_).astype(np.float64))


def test_model_without_a_map_unchanged(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.random((200, 4), dtype=np.float32)
    y = (X[:, 0] > 0.5).astype(np.float32)
    model = mojolearn.GradientBoosting(loss="Logloss", n_estimators=6, device="cpu").fit(X, y)
    path = str(tmp_path / "plain.npz")
    model.save(path)
    host = _hgc_host(path)
    assert not hasattr(host, "category_map_")
    np.testing.assert_array_equal(np.asarray(host.predict(X)), np.asarray(model.predict(X)))
    np.testing.assert_array_equal(np.asarray(host.predict_proba(X)), np.asarray(model.predict_proba(X)))
    assert host.digest_ == model.digest_


def test_logloss_proba_matches(tmp_path):
    cat, _, num, y = _hgc_data()
    yb = (y > 1.5).astype(np.float32)
    model = mojolearn.GradientBoosting(loss="Logloss", n_estimators=6, max_depth=3, cat_features=[0], device="cpu")
    model.fit(_hgc_rows(cat, num), yb)
    path = str(tmp_path / "logloss.npz")
    model.save(path)
    qc, qn = _hgc_queries()
    X = _hgc_rows(qc, qn)
    np.testing.assert_array_equal(np.asarray(_hgc_host(path).predict_proba(X)), np.asarray(model.predict_proba(X)))


@pytest.mark.parametrize("options", [
    {"max_ctr_complexity": 2, "boosting_type": "Plain"},
    {"boosting_type": "Ordered"},
])
def test_ctr_models_match(tmp_path, options):
    # 40 categories per column: CTR columns, not one-hot; two categorical
    # columns so max_ctr_complexity 2 can combine them.
    cat, cat2, num, y = _hgc_data(_HGC_MANY)
    X = [[cat[i], cat2[i], float(num[i])] for i in range(_HGC_N)]
    model = mojolearn.GradientBoosting(n_estimators=8, max_depth=3, cat_features=[0, 1], device="cpu", **options)
    model.fit(X, y)
    assert "ctr_" in model.model_, "the fit is meant to build CTR columns (40 categories)"
    path = str(tmp_path / "ctr.npz")
    model.save(path)
    Q = X[:20] + [["unseen", None, 0.5], [None, "c01", -1.0]]
    np.testing.assert_array_equal(np.asarray(_hgc_host(path).predict(Q)), np.asarray(model.predict(Q)))
