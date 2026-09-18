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
from mojolearn.ensemble import _group_sizes

REFUSAL = "group_id is read only by the querywise and pairwise losses"


def test_runs_of_uneven_size_including_one():
    assert _group_sizes([5, 5, 5, 9, "x", "x"], 6) == [3, 1, 2]
    assert _group_sizes(["a"], 1) == [1]
    assert _group_sizes([0, 1, 2, 3], 4) == [1, 1, 1, 1]


def test_integer_and_string_spellings_are_one_group():
    # CatBoost hashes ToString<i64>(id) for an integer and the string itself
    # for a string, so 7 and "7" are the same group id
    assert _group_sizes([7, "7", np.int64(7), b"7"], 4) == [4]


def test_numpy_integer_arrays():
    ids = np.array([3, 3, 1, 1, 1, 2], dtype=np.int32)
    assert _group_sizes(ids, 6) == [2, 3, 1]
    assert _group_sizes(np.array(["q1", "q1", "q2"]), 3) == [2, 1]


@pytest.mark.parametrize("bad", [1.0, np.float32(2.0), True, None, (1,)])
def test_unsuitable_ids_are_refused_in_their_words(bad):
    with pytest.raises(ValueError, match=r"should be string or integral type"):
        _group_sizes([1, bad], 2)


def test_split_group_is_refused():
    with pytest.raises(ValueError, match="group Ids are not consecutive"):
        _group_sizes([1, 1, 2, 1], 4)
    with pytest.raises(ValueError, match="group Ids are not consecutive"):
        _group_sizes(["a", 7, "a"], 3)


def test_length_must_match_rows():
    with pytest.raises(ValueError, match="Length of group_id=3 and length of data=4"):
        _group_sizes([1, 1, 2], 4)
    with pytest.raises(ValueError, match="must be array like"):
        _group_sizes(5, 1)


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


# --- the run-based grouping (lane/gbdt-group-sizes, 2026-09-17) -------------
# `_group_sizes_by_runs` keys one id per run; `_group_sizes_rowwise` is the
# definition. The first two tests pin WHICH path answers, so the differential
# below cannot pass by always deferring to the row walk.

def _outcome(fn, *args):
    try:
        return ("ok", fn(*args))
    except Exception as exc:  # the type and the words are both the contract
        return (type(exc).__name__, str(exc))


def test_run_path_answers_plain_ids():
    from mojolearn.ensemble import _group_sizes_by_runs
    assert _group_sizes_by_runs([4, 4, 4, 2, 2, 9]) == [3, 2, 1]
    assert _group_sizes_by_runs(["a", "a", "b"]) == [2, 1]
    assert _group_sizes_by_runs([b"q", b"q"]) == [2]
    # adjacent runs that spell one key merge, as the row walk merges them
    assert _group_sizes_by_runs([7, 7, "7", b"7", 8]) == [4, 1]
    assert _group_sizes_by_runs([]) == []


@pytest.mark.parametrize("ids", [
    [5, 5.0],            # equal under ==, and the float is refused
    [1, True],           # equal under ==, and the bool is refused
    [True, True],        # a bool is not an int here
    [1.5],
    [None],
    [3, 3, 4, 3],        # a repeated group: the row walk words the refusal
    [np.int64(3), 3],    # a list holding a NumPy scalar: not exactly int
    [np.array([1, 2]), 1],  # == returns an array; its truth value raises
])
def test_run_path_defers(ids):
    from mojolearn.ensemble import _group_sizes_by_runs
    assert _group_sizes_by_runs(ids) is None


def test_runs_equal_the_row_walk_on_random_ids():
    import random
    from mojolearn.ensemble import _group_sizes_rowwise
    rng = random.Random(20260917)
    spell = [lambda v: v, lambda v: str(v), lambda v: str(v).encode("ascii")]
    odd = [5.0, True, None, (1,), 2.5]
    deferred = answered = 0
    for case in range(400):
        ids = []
        for group in rng.sample(range(40), rng.randint(1, 12)):
            mixed = rng.random() < 0.3
            one = rng.choice(spell)
            for _ in range(rng.randint(1, 6)):
                ids.append((rng.choice(spell) if mixed else one)(group))
        if rng.random() < 0.25:     # a refused id somewhere
            ids.insert(rng.randrange(len(ids) + 1), rng.choice(odd))
        if rng.random() < 0.25:     # a group split into two runs
            ids.append(ids[0])
        want = _outcome(_group_sizes_rowwise, ids)
        assert _outcome(_group_sizes, ids, len(ids)) == want, ids
        if want[0] == "ok":
            answered += 1
        else:
            deferred += 1
    assert answered > 100 and deferred > 100, (answered, deferred)


def test_a_two_million_id_pool_is_grouped_by_runs():
    from mojolearn.ensemble import _group_sizes_by_runs
    ids = [q for q in range(20000) for _ in range(100)]
    assert _group_sizes_by_runs(ids) == [100] * 20000
