"""group_id in the GradientBoosting pool (lane/gbdt-learning-to-rank, stage 1).

The grouping follows CatBoost's Pool: ids are strings or integers compared by
their byte spelling (`_catboost.pyx:2171-2196`), a group's rows are
consecutive (`libs/data/objects.cpp:60-87`), and every loss this
implementation trains refuses a grouping BY NAME until a querywise loss
lands. The refusal is raised inside the binding (`gbdt/train.mojo::train`
on a GPU install, `bindings/_mojolearn_gbdt_host.mojo::gbdt_fit_binding` on
a CPU-only one), so the fit tests below prove the sizes cross the ABI tail.
"""
import numpy as np
import pytest

from mojolearn import GradientBoosting
from mojolearn._cpu_reference import reference_training

REFUSAL = "group_id is read only by the querywise and pairwise losses"


def _small():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((64, 3)).astype(np.float32)
    y = rng.integers(0, 3, size=64).astype(np.float32)
    group = np.repeat(np.arange(11), [1, 2, 9, 1, 5, 6, 4, 8, 12, 7, 9])
    return X, y, group


@reference_training()
@pytest.mark.parametrize("loss", ["RMSE", "Logloss"])
def test_every_trained_loss_refuses_group_id_by_name(loss):
    X, y, group = _small()
    target = (y > 1).astype(np.float32) if loss == "Logloss" else y
    with pytest.raises(Exception, match=REFUSAL):
        GradientBoosting(n_estimators=2, max_depth=2, loss=loss).fit(X, target, group_id=group)


@reference_training()
def test_split_group_never_reaches_the_binding():
    X, y, group = _small()
    group = group.copy()
    group[-1] = 0
    with pytest.raises(ValueError, match="group Ids are not consecutive"):
        GradientBoosting(n_estimators=2, max_depth=2).fit(X, y, group_id=group)


@reference_training()
def test_subgroup_id_and_pairs_are_refused_by_name():
    X, y, group = _small()
    with pytest.raises(NotImplementedError, match="subgroup_id"):
        GradientBoosting(n_estimators=2).fit(X, y, subgroup_id=group)
    with pytest.raises(NotImplementedError, match="pairs"):
        GradientBoosting(n_estimators=2).fit(X, y, pairs=[(0, 1)])


def test_a_fit_without_group_id_packs_the_old_layout():
    params = GradientBoosting()._params(32, 4, 0)
    assert len(params) == 35


# --- the grouping through the public fit: the sizes a group_id spells are
# what the querywise loss trains on, so equal groupings give one model.

def _ranker():
    return GradientBoosting(n_estimators=2, max_depth=2, loss="QueryRMSE",
                            bootstrap_type="No", random_strength=0.0)


def _ranked(group_id, X=None, y=None):
    if X is None:
        X, y, _ = _small()
    return np.asarray(_ranker().fit(X, y, group_id=group_id).predict(X)).tobytes()


@reference_training()
def test_spellings_of_one_grouping_give_one_model():
    X, y, group = _small()
    base = _ranked(group, X, y)
    # an int array, a list of ints, strings, bytes and NumPy int32 are one
    # grouping; CatBoost hashes ToString<i64>(id) for an integer, so 7 and
    # "7" are the same group
    spellings = [
        group.tolist(),
        [str(v) for v in group.tolist()],
        [str(v).encode("ascii") for v in group.tolist()],
        group.astype(np.int32),
        [v if i % 2 else str(v) for i, v in enumerate(group.tolist())],
    ]
    for ids in spellings:
        assert _ranked(ids, X, y) == base
    # a different grouping trains a different model
    assert _ranked(np.zeros(len(group), np.int64), X, y) != base


@reference_training()
@pytest.mark.parametrize("bad", [1.0, np.float32(2.0), True, None, (1,)])
def test_unsuitable_ids_are_refused_in_their_words(bad):
    X, y, group = _small()
    ids = group.tolist()
    ids[1] = bad
    with pytest.raises(ValueError, match=r"should be string or integral type"):
        _ranker().fit(X, y, group_id=ids)


@reference_training()
def test_split_and_length_refusals():
    X, y, group = _small()
    n = len(group)
    ids = ["a"] * n
    ids[1] = 7
    with pytest.raises(ValueError, match="group Ids are not consecutive"):
        _ranker().fit(X, y, group_id=ids)
    with pytest.raises(ValueError, match=f"Length of group_id={n - 1} and length of data={n}"):
        _ranker().fit(X, y, group_id=group[:-1])
    with pytest.raises(ValueError, match="must be array like"):
        _ranker().fit(X, y, group_id=5)
