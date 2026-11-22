# SPDX-License-Identifier: Apache-2.0
"""Python boundary tests: `HostGBDT` applies a saved model's category map
(fx-hostgbdt, 2026-11-21).

A `GradientBoosting(cat_features=...)` file holds its category map; the CPU
twin `HostGBDT` encodes raw categorical input through it with the same
admission and apply functions (bindings/gbdt_category_unpack.mojo), so the
same file gives the same bits from either class, on every accepted input
form, with unseen values and None, and refuses in the same words. The
tree-CTR (max_ctr_complexity 2) and Ordered CTR models carry their tables in
the model text, which the host predict expands. Small fits only (a few
trees, CPU); nothing here is a measurement.

The Plain fits use the categorical arm the CPU gbdt binding trains
(SymmetricTree, Logloss, no bootstrap, no score noise:
gbdt/mojolearn_gbdt_host_run.mojo refuses bootstrap_type and cat_features
outside SymmetricTree with Logloss by name); the Ordered CTR fit keeps its
defaults, which the CPU Ordered arm trains.
"""
import pickle

import numpy as np
import pytest

mojolearn = pytest.importorskip("mojolearn")

N = 240
CATS = ["red", "green", "blue", "teal"]
MANY = ["c%02d" % i for i in range(40)]
CPU_CTR = {"loss": "Logloss", "bootstrap_type": "No", "random_strength": 0.0}


def _data(cats=CATS):
    rng = np.random.default_rng(5)
    cat = [cats[(7 * i) % len(cats)] for i in range(N)]
    cat2 = [cats[(3 * i + 1) % len(cats)] for i in range(N)]
    num = rng.normal(size=N).astype(np.float64)
    y = np.array([float(cats.index(cat[i]) % 4) + 0.1 * num[i] for i in range(N)], dtype=np.float32)
    return cat, cat2, num, y


def _rows(cat, num):
    return [[cat[i], float(num[i])] for i in range(len(cat))]


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    cat, _, num, y = _data()
    model = mojolearn.GradientBoosting(n_estimators=8, max_depth=3, cat_features=[0], device="cpu", **CPU_CTR)
    model.fit(_rows(cat, num), (y > 1.5).astype(np.float32))
    path = str(tmp_path_factory.mktemp("hostgbdt") / "cat.npz")
    model.save(path)
    return model, path


def _host(path):
    return mojolearn.HostGBDT.from_file(path)


def _queries():
    cat, _, num, _ = _data()
    test_cat = cat[:20] + ["violet", None, "red", 3]
    test_num = list(num[:24])
    return test_cat, test_num


def test_host_matches_model_on_list_rows_with_unseen_and_none(saved):
    model, path = saved
    cat, num = _queries()
    X = _rows(cat, num)
    np.testing.assert_array_equal(np.asarray(_host(path).predict(X)), np.asarray(model.predict(X)))


def test_host_matches_model_on_every_input_form(saved):
    model, path = saved
    host = _host(path)
    cat, _, num, _ = _data()
    cat, num = cat[:30], num[:30]
    expected = np.asarray(model.predict(_rows(cat, num)))
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


def test_host_refuses_in_the_models_words(saved):
    model, path = saved
    host = _host(path)
    cat, num = _queries()
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


def test_digest_equal_and_map_kept(saved, tmp_path):
    model, path = saved
    host = _host(path)
    assert host.digest_ == model.digest_
    np.testing.assert_array_equal(np.asarray(host.category_map_), np.asarray(model.category_map_))
    restored = pickle.loads(pickle.dumps(host))
    cat, num = _queries()
    X = _rows(cat, num)
    np.testing.assert_array_equal(np.asarray(restored.predict(X)), np.asarray(model.predict(X)))


def test_constructor_takes_the_map(saved):
    model, path = saved
    host = mojolearn.HostGBDT(loss=model.loss, text=model.model_, n_features_in=model.n_features_in_,
                              approx_dim=model.approx_dim_, n_classes=model.n_classes_, bias=model.bias_,
                              category_map=model.category_map_)
    cat, num = _queries()
    X = _rows(cat, num)
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
    host = _host(path)
    assert not hasattr(host, "category_map_")
    np.testing.assert_array_equal(np.asarray(host.predict(X)), np.asarray(model.predict(X)))
    np.testing.assert_array_equal(np.asarray(host.predict_proba(X)), np.asarray(model.predict_proba(X)))
    assert host.digest_ == model.digest_


def test_logloss_proba_matches(tmp_path):
    cat, _, num, y = _data()
    yb = (y > 1.5).astype(np.float32)
    model = mojolearn.GradientBoosting(n_estimators=6, max_depth=3, cat_features=[0], device="cpu", **CPU_CTR)
    model.fit(_rows(cat, num), yb)
    path = str(tmp_path / "logloss.npz")
    model.save(path)
    qc, qn = _queries()
    X = _rows(qc, qn)
    np.testing.assert_array_equal(np.asarray(_host(path).predict_proba(X)), np.asarray(model.predict_proba(X)))


@pytest.mark.parametrize("options", [
    {"max_ctr_complexity": 2, "boosting_type": "Plain", **CPU_CTR},
    {"boosting_type": "Ordered"},
])
def test_ctr_models_match(tmp_path, options):
    # 40 categories per column: CTR columns, not one-hot; two categorical
    # columns so max_ctr_complexity 2 can combine them.
    cat, cat2, num, y = _data(MANY)
    X = [[cat[i], cat2[i], float(num[i])] for i in range(N)]
    model = mojolearn.GradientBoosting(n_estimators=8, max_depth=3, cat_features=[0, 1], device="cpu", **options)
    model.fit(X, (y > 1.5).astype(np.float32) if options.get("loss") == "Logloss" else y)
    assert "ctr_" in model.model_, "the fit is meant to build CTR columns (40 categories)"
    path = str(tmp_path / "ctr.npz")
    model.save(path)
    Q = X[:20] + [["unseen", None, 0.5], [None, "c01", -1.0]]
    np.testing.assert_array_equal(np.asarray(_host(path).predict(Q)), np.asarray(model.predict(Q)))
