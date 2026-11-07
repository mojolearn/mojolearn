# SPDX-License-Identifier: Apache-2.0
"""Search, splitter, scorer and fixed-owner records of lane ne2-b-search-workers.

Rows N056, N066, N067, N069, N070, N074, N086 of
docs/plans/native_entry_points_after_0866/owed.tsv. The bits of every split,
permutation and score are the identity lanes' (`x-metrics-splitters`,
`x-metrics-permutation-test-score`, `x-metrics-search`, `cross-val`); these
tests check the contracts the changes rely on: determinism, the split
invariants, the records being dropped when the state they prove changes, and
the DeviceRows flags.

Runs on a box with a built wheel (never on the laptop).
"""
import pytest

np = pytest.importorskip("numpy")
mojolearn = pytest.importorskip("mojolearn")

from mojolearn import model_selection as ms  # noqa: E402


def _labels(n=60, k=3, seed=5):
    rng = np.random.default_rng(seed)
    return rng.integers(0, k, size=n).astype("i8")


def _pairs(splits):
    return [(np.asarray(tr).tobytes(), np.asarray(te).tobytes()) for tr, te in splits]


# ---------------------------------------------------------------- N086


@pytest.mark.parametrize("test_size", [0.25, 0.5, 9])
def test_stratified_shuffle_split_one_program_invariants(test_size):
    y = _labels()
    x = np.zeros((len(y), 2), dtype="f4")
    splitter = ms.StratifiedShuffleSplit(n_splits=4, test_size=test_size, random_state=3)
    first = list(splitter.split(x, y))
    again = list(splitter.split(x, y))
    assert _pairs(first) == _pairs(again)
    n_test = int(np.ceil(test_size * len(y))) if isinstance(test_size, float) else test_size
    for train, test in first:
        train, test = np.asarray(train), np.asarray(test)
        assert train.dtype == np.int64 and test.dtype == np.int64
        assert len(test) == n_test and len(train) == len(y) - n_test
        assert len(np.intersect1d(train, test)) == 0
        assert len(np.unique(train)) == len(train) and len(np.unique(test)) == len(test)
        # every class is on both sides (approximate_mode keeps each class)
        assert set(y[train]) == set(y) and set(y[test]) == set(y)
    # consecutive splits differ: each split consumes its own draws
    assert _pairs(first[:1]) != _pairs(first[1:2])


def test_train_test_split_stratify_is_deterministic():
    y = _labels(n=41, k=4, seed=9)
    x = np.arange(len(y) * 2, dtype="f4").reshape(len(y), 2)
    a = ms.train_test_split(x, y, test_size=0.3, stratify=y, random_state=7)
    b = ms.train_test_split(x, y, test_size=0.3, stratify=y, random_state=7)
    for left, right in zip(a, b):
        assert np.asarray(left).tobytes() == np.asarray(right).tobytes()


# ---------------------------------------------------------------- N056


def test_permutation_test_score_groups_is_deterministic():
    from mojolearn import linear_model
    rng = np.random.default_rng(2)
    x = rng.normal(size=(48, 3)).astype("f4")
    y = (x[:, 0] - x[:, 1]).astype("f4")
    groups = np.repeat(np.arange(6), 8).astype("i8")
    args = dict(groups=groups, cv=ms.KFold(3), n_permutations=5, random_state=4)
    one = ms.permutation_test_score(linear_model.Ridge(alpha=1.0), x, y, **args)
    two = ms.permutation_test_score(linear_model.Ridge(alpha=1.0), x, y, **args)
    assert one[0] == two[0] and one[2] == two[2]
    assert np.asarray(one[1]).tobytes() == np.asarray(two[1]).tobytes()


# ---------------------------------------------------------------- N070, N074


@pytest.mark.parametrize("make", [
    lambda: ms.KFold(3), lambda: ms.KFold(3, shuffle=True, random_state=1), lambda: ms.StratifiedKFold(3),
    lambda: ms.GroupKFold(3), lambda: ms.StratifiedGroupKFold(3), lambda: ms.TimeSeriesSplit(3),
    lambda: ms.LeaveOneOut(), lambda: ms.LeavePOut(2), lambda: ms.LeaveOneGroupOut(), lambda: ms.LeavePGroupsOut(2),
    lambda: ms.ShuffleSplit(3, random_state=0), lambda: ms.GroupShuffleSplit(3, random_state=0),
    lambda: ms.StratifiedShuffleSplit(3, random_state=0), lambda: ms.RepeatedKFold(n_splits=2, n_repeats=2, random_state=0),
    lambda: ms.RepeatedStratifiedKFold(n_splits=2, n_repeats=2, random_state=0),
])
def test_every_splitter_kind_splits_and_counts(make):
    # N070: the class code replaces the per-name tests. Every connected
    # splitter still dispatches to its own split, and get_n_splits agrees
    # with the number of splits it makes.
    y = np.array([0, 1] * 6, dtype="i8")
    groups = np.repeat(np.arange(4), 3).astype("i8")
    x = np.zeros((len(y), 2), dtype="f4")
    splitter = make()
    splits = list(splitter.split(x, y, groups))
    assert len(splits) == splitter.get_n_splits(x, y, groups)
    assert _pairs(splits) == _pairs(list(make().split(x, y, groups)))


@pytest.mark.parametrize("groups", [
    np.array([3, 1, 3, 2, 1, 7], dtype="i8"), np.array([3, 1, 3, 2, 1, 7], dtype="i4"),
    [3, 1, 3, 2, 1, 7], np.array([0.5, 1.0, 0.5, 2.0, 1.0, 1.0]), ["a", "b", "a", "c", "b", "b"],
])
def test_group_count_matches_the_distinct_labels(groups):
    # N074: an integer vector is counted from its buffer; every other kind
    # keeps the listed set. Both equal the number of distinct groups.
    distinct = len(set(np.asarray(groups).tolist()))
    x = np.zeros((6, 1), dtype="f4")
    assert ms.LeaveOneGroupOut().get_n_splits(x, None, groups) == distinct
    assert ms.LeavePGroupsOut(2).get_n_splits(x, None, groups) == distinct * (distinct - 1) // 2


# ---------------------------------------------------------------- N067


def test_scorer_class_record_drops_on_class_change(monkeypatch):
    # The scorer class rows are checked once per class state; a changed hook
    # on the class is refused on the next score (type watchers; every call
    # on Python < 3.12).
    from mojolearn import linear_model
    rng = np.random.default_rng(1)
    x = rng.normal(size=(30, 2)).astype("f4")
    y = (x[:, 0] + x[:, 1]).astype("f4")
    first = ms.cross_val_score(linear_model.Ridge(), x, y, cv=3, scoring="r2")
    second = ms.cross_val_score(linear_model.Ridge(), x, y, cv=3, scoring="r2")
    assert np.asarray(first).tobytes() == np.asarray(second).tobytes()
    monkeypatch.setattr(ms._Scorer, "__repr__", lambda self: "changed")
    with pytest.raises(TypeError):
        ms.cross_val_score(linear_model.Ridge(), x, y, cv=3, scoring="r2")


def test_scorer_field_keys_record_drops_on_dictionary_change():
    from mojolearn import linear_model
    rng = np.random.default_rng(3)
    x = rng.normal(size=(30, 2)).astype("f4")
    y = (x[:, 0] > 0).astype("i8")
    scorer = ms.get_scorer("accuracy")
    ms.cross_val_score(linear_model.LogisticRegression(max_iter=20), x, y, cv=3, scoring=scorer)
    vars(scorer)[1] = "a non-str key"
    with pytest.raises(TypeError):
        ms.cross_val_score(linear_model.LogisticRegression(max_iter=20), x, y, cv=3, scoring=scorer)


# ---------------------------------------------------------------- N069


def test_fixed_owner_record_drops_on_assignment():
    # The fixed-owner admission is kept per model state; replacing a fitted
    # field re-admits it on the next call.
    from mojolearn.naive_bayes import GaussianNB
    rng = np.random.default_rng(4)
    x = rng.normal(size=(40, 3)).astype("f4")
    y = (x[:, 0] > 0).astype("i8")
    model = GaussianNB().fit(x, y)
    a = model.predict(x)
    b = model.predict(x)
    assert np.asarray(a).tobytes() == np.asarray(b).tobytes()
    vars(model)["classes_"] = object()
    with pytest.raises(Exception):
        model.predict(x)


# ---------------------------------------------------------------- DeviceRows flags


@pytest.mark.parametrize("name", ["LinearRegression", "Ridge", "MaxAbsScaler", "RobustScaler"])
def test_device_rows_flag_is_on(name):
    from mojolearn import linear_model, _expansion_prep as prep
    cls = getattr(linear_model, name, None) or getattr(prep, name)
    registry = ms._NATIVE_CONTEXT["family_msel_estimators"]
    if cls not in registry:
        pytest.skip(f"{name} has no native model-selection plan in this build")
    assert registry[cls]["device_rows"] is True


@pytest.mark.parametrize("name", ["LinearRegression", "Ridge"])
def test_device_rows_folds_score_like_the_host_path(name):
    from mojolearn import linear_model
    rng = np.random.default_rng(11)
    x = rng.normal(size=(60, 5)).astype("f4")
    y = (2 * x[:, 0] - x[:, 1]).astype("f4")
    plan = ms._NATIVE_CONTEXT["family_msel_estimators"][getattr(linear_model, name)]
    try:
        plan["device_rows"] = False
        host = ms.cross_val_score(getattr(linear_model, name)(), x, y, cv=5)
        plan["device_rows"] = True
        door = ms.cross_val_score(getattr(linear_model, name)(), x, y, cv=5)
    finally:
        plan["device_rows"] = True
    assert np.asarray(door, dtype="f8").tobytes() == np.asarray(host, dtype="f8").tobytes()
