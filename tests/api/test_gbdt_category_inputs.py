# SPDX-License-Identifier: Apache-2.0
"""Python boundary tests: the input forms `GradientBoosting(cat_features=...)`
admits for its categorical columns (bindings/gbdt_category_unpack.mojo), at
fit and at predict, and the refusals by name.

Every accepted form of the same data must give the same predictions: the
category of a value does not depend on the container it arrived in. Small
fits only (a few trees, CPU); nothing here is a measurement.
"""
import numpy as np
import pytest

mojolearn = pytest.importorskip("mojolearn")

N = 240
CATS = ["red", "green", "blue", "teal"]


def _data():
    rng = np.random.default_rng(3)
    cat = [CATS[i % 4] for i in range(N)]
    num = rng.normal(size=N).astype(np.float64)
    y = np.array([float(i % 4) + 0.1 * num[i] for i in range(N)], dtype=np.float32)
    return cat, num, y


def _model():
    return mojolearn.GradientBoosting(n_estimators=8, max_depth=3, cat_features=[0], device="cpu")


def _fit_predict(X, y, X_test=None):
    model = _model().fit(X, y)
    return model.predict(X if X_test is None else X_test)


@pytest.fixture(scope="module")
def reference():
    cat, num, y = _data()
    rows = [[cat[i], float(num[i])] for i in range(N)]
    return _fit_predict(rows, y)


def test_object_array_matches_list_rows(reference):
    cat, num, y = _data()
    X = np.empty((N, 2), dtype=object)
    X[:, 0] = cat
    X[:, 1] = num
    np.testing.assert_array_equal(_fit_predict(X, y), reference)
    # a strided object view is read through its own strides
    wide = np.empty((N, 4), dtype=object)
    wide[:, 0] = cat
    wide[:, 2] = num
    np.testing.assert_array_equal(_fit_predict(wide[:, ::2], y), reference)


def test_unicode_and_bytes_arrays_with_numeric_text(reference):
    cat, num, y = _data()
    # repr round-trips a float64, so the parsed column holds the same values
    U = np.array([[cat[i], repr(float(num[i]))] for i in range(N)])
    assert U.dtype.kind == "U"
    np.testing.assert_array_equal(_fit_predict(U, y), reference)
    S = U.astype("S")
    np.testing.assert_array_equal(_fit_predict(S, y), reference)


def test_unicode_array_numeric_column_refused_in_words():
    cat, num, y = _data()
    U = np.array([[cat[i], "x" if i == 7 else "1.0"] for i in range(N)])
    with pytest.raises(ValueError, match=r"X\[7\]\[1\] is not a number"):
        _model().fit(U, y)


pd = pytest.importorskip("pandas")


def _frame(cat, num, dtype):
    df = pd.DataFrame({"c": pd.Series(cat, dtype=dtype), "x": num})
    return df


@pytest.mark.parametrize("dtype", [object, "category"])
def test_pandas_columns_match_list_rows(reference, dtype):
    cat, num, y = _data()
    np.testing.assert_array_equal(_fit_predict(_frame(cat, num, dtype), y), reference)


def test_pandas_python_string_column(reference):
    cat, num, y = _data()
    try:
        dtype = pd.StringDtype("python")
    except TypeError:
        pytest.skip("this pandas has no Python-backed string dtype")
    np.testing.assert_array_equal(_fit_predict(_frame(cat, num, dtype), y), reference)


def test_pandas_category_codes_are_not_the_categories(reference):
    # the same values with categories in another order: other codes, same categories
    cat, num, y = _data()
    df = _frame(cat, num, pd.CategoricalDtype(categories=list(reversed(CATS))))
    np.testing.assert_array_equal(_fit_predict(df, y), reference)


def test_pandas_missing_is_one_category():
    cat, num, y = _data()
    with_missing = [None if i % 9 == 0 else cat[i] for i in range(N)]
    rows = [[with_missing[i], float(num[i])] for i in range(N)]
    expected = _fit_predict(rows, y)
    as_category = _frame(with_missing, num, "category")  # None -> code -1
    np.testing.assert_array_equal(_fit_predict(as_category, y), expected)
    as_na = _frame([pd.NA if v is None else v for v in with_missing], num, object)
    np.testing.assert_array_equal(_fit_predict(as_na, y), expected)


def test_predict_maps_unseen_values_through_the_stored_map():
    cat, num, y = _data()
    model = _model().fit(_frame(cat, num, "category"), y)
    test = _frame(["red", "violet", "blue"], num[:3], "category")
    got = model.predict(test)
    rows = model.predict([["red", float(num[0])], ["violet", float(num[1])], ["blue", float(num[2])]])
    np.testing.assert_array_equal(got, rows)


def test_numeric_category_column_and_numeric_feature():
    cat, num, y = _data()
    codes = [i % 4 for i in range(N)]
    plain = pd.DataFrame({"c": codes, "x": num, "z": np.arange(N, dtype=np.int64) % 5})
    as_category = plain.astype({"c": "category", "z": "category"})  # z is a numeric feature
    np.testing.assert_array_equal(_fit_predict(as_category, y), _fit_predict(plain, y))


def _arrow_dtype(storage):
    pytest.importorskip("pyarrow")
    try:
        return pd.StringDtype(storage)
    except (TypeError, ImportError):
        pytest.skip("this pandas has no " + storage + " string dtype")


def test_arrow_string_column_matches_list_rows(reference):
    # pandas 2 ``string[pyarrow]``; pandas 3's default ``str`` with pyarrow
    cat, num, y = _data()
    df = _frame(cat, num, _arrow_dtype("pyarrow"))
    np.testing.assert_array_equal(_fit_predict(df, y), reference)


def test_arrow_large_string_and_chunked(reference):
    pa = pytest.importorskip("pyarrow")
    cat, num, y = _data()
    half = N // 2
    for typ in (pa.string(), pa.large_string()):
        chunked = pa.chunked_array([pa.array(cat[:half], type=typ), pa.array(cat[half:], type=typ)])
        column = pd.Series(pd.arrays.ArrowExtensionArray(chunked))
        df = pd.DataFrame({"c": column, "x": num})
        np.testing.assert_array_equal(_fit_predict(df, y), reference)


def test_arrow_sliced_chunk_offsets(reference):
    # a chunk that is a slice of a larger array: its offset is not 0
    pa = pytest.importorskip("pyarrow")
    cat, num, y = _data()
    padded = pa.array(["pad"] * 7 + cat + ["pad"] * 3).slice(7, N)
    column = pd.Series(pd.arrays.ArrowExtensionArray(pa.chunked_array([padded])))
    np.testing.assert_array_equal(_fit_predict(pd.DataFrame({"c": column, "x": num}), y), reference)


def test_arrow_dictionary_column(reference):
    pa = pytest.importorskip("pyarrow")
    cat, num, y = _data()
    encoded = pa.array(cat).dictionary_encode()
    column = pd.Series(pd.arrays.ArrowExtensionArray(pa.chunked_array([encoded])))
    np.testing.assert_array_equal(_fit_predict(pd.DataFrame({"c": column, "x": num}), y), reference)


def test_arrow_missing_is_one_category():
    cat, num, y = _data()
    with_missing = [None if i % 9 == 0 else cat[i] for i in range(N)]
    rows = [[with_missing[i], float(num[i])] for i in range(N)]
    expected = _fit_predict(rows, y)
    df = _frame(with_missing, num, _arrow_dtype("pyarrow"))
    np.testing.assert_array_equal(_fit_predict(df, y), expected)


def test_arrow_categories_of_a_categorical(reference):
    # pandas 3 keeps str categories in an Arrow-backed Index
    cat, num, y = _data()
    dtype = pd.CategoricalDtype(pd.Index(CATS, dtype=_arrow_dtype("pyarrow")))
    np.testing.assert_array_equal(_fit_predict(_frame(cat, num, dtype), y), reference)


def test_arrow_fit_object_predict_agree(reference):
    # the same text is the same category whatever storage predict receives
    cat, num, y = _data()
    model = _model().fit(_frame(cat, num, _arrow_dtype("pyarrow")), y)
    np.testing.assert_array_equal(model.predict(_frame(cat, num, object)), reference)


def test_arrow_column_outside_cat_features_refused():
    cat, num, y = _data()
    df = pd.DataFrame({"c": pd.Series(cat, dtype=object),
                       "t": pd.Series(cat, dtype=_arrow_dtype("pyarrow")), "x": num})
    with pytest.raises(TypeError, match="outside cat_features"):
        _model().fit(df, y)


def test_arrow_numeric_type_refused_by_name():
    pa = pytest.importorskip("pyarrow")
    cat, num, y = _data()
    column = pd.Series(pd.arrays.ArrowExtensionArray(pa.chunked_array([pa.array(list(range(N)))])))
    with pytest.raises(TypeError, match="Int64Array"):
        _model().fit(pd.DataFrame({"c": column, "x": num}), y)


def test_text_in_numeric_pandas_column_refused():
    cat, num, y = _data()
    df = pd.DataFrame({"c": cat, "x": [str(v) for v in num]}, dtype=object)
    with pytest.raises(TypeError, match="only the cat_features columns may hold text"):
        _model().fit(df, y)
