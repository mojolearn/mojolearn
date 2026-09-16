# SPDX-License-Identifier: Apache-2.0
"""Fold metadata matches sklearn; built-in CV never needs its dependencies."""
import os
import subprocess
import sys
import textwrap

import pytest

from mojolearn.model_selection import _clone, _default_folds


@pytest.mark.parametrize('labels', [
    [0, 1] * 11,
    ['z'] * 8 + ['a'] * 7 + ['m'] * 6,
    [2, 0, 1, 1, 2, 0] * 4,
    [0] * 3 + [1] * 9,
])
@pytest.mark.parametrize('splits', [2, 3])
def test_stratified_metadata_matches_reference(labels, splits):
    reference = pytest.importorskip('sklearn.model_selection')
    expected = list(reference.StratifiedKFold(splits).split(labels, labels))
    actual = list(_default_folds(labels, splits, True))
    assert actual == [(train.tolist(), test.tolist()) for train, test in expected]


def test_kfold_metadata_matches_reference():
    reference = pytest.importorskip('sklearn.model_selection')
    for n in [7, 11, 22]:
        for splits in [2, 3, 5]:
            expected = reference.KFold(splits).split(range(n))
            assert list(_default_folds(list(range(n)), splits, False)) == [
                (train.tolist(), test.tolist()) for train, test in expected]


def test_pipeline_clone_keeps_external_protocol():
    pipeline = pytest.importorskip('sklearn.pipeline')
    preprocessing = pytest.importorskip('sklearn.preprocessing')
    original = pipeline.Pipeline([('scale', preprocessing.StandardScaler())])
    cloned = _clone(original)
    assert cloned is not original
    assert cloned.steps[0][1] is not original.steps[0][1]
    assert cloned.get_params()['scale__with_mean'] is True


def test_default_and_explicit_cv_without_numpy_or_sklearn():
    # Fresh process guarantees an already-imported dependency cannot mask reach.
    program = textwrap.dedent('''
        import builtins, ctypes
        original_import = builtins.__import__
        def blocked(name, *args, **kwargs):
            if name.split('.')[0] in ('numpy', 'sklearn'):
                raise AssertionError('unexpected dependency: ' + name)
            return original_import(name, *args, **kwargs)
        builtins.__import__ = blocked
        from mojolearn import Array
        from mojolearn import _buffer
        from mojolearn.model_selection import cross_val_score
        calls = []
        def gather(src, dst, indices, source_rows, output_rows, row_bytes):
            for output, source in enumerate((ctypes.c_int64 * output_rows).from_address(indices)):
                assert 0 <= source < source_rows
                ctypes.memmove(dst + output * row_bytes, src + source * row_bytes, row_bytes)
        _buffer._NATIVE['gather_rows_bytes'] = gather
        class Estimator:
            _estimator_type = 'classifier'
            def __init__(self, setting=None): self.setting = setting
            def get_params(self, deep=True): return {'setting': self.setting}
            def fit(self, X, y):
                assert not hasattr(self, 'fitted_')
                assert self.setting == {'nested': [1, 2]}
                self.setting['nested'].append(3)
                self.fitted_ = True
                calls.append((X.tolist(), y.tolist()))
                return self
            def score(self, X, y):
                assert self.fitted_
                return 0.75
        X = Array.from_list([[float(i), float(i+100)] for i in range(12)], '<f4')
        y = Array.from_list([0, 1] * 6, '<i4')
        model = Estimator({'nested': [1, 2]})
        assert cross_val_score(model, X, y).tolist() == [0.75] * 5
        assert model.setting == {'nested': [1, 2]}
        assert not hasattr(model, 'fitted_')
        assert cross_val_score(model, X, y, cv=[([0, 1, 2, 3], [4, 5])]).tolist() == [0.75]
        class Splitter:
            def split(self, X, y, groups):
                assert groups == ['a'] * 12
                yield [0, 1], [8, 9]
        assert cross_val_score(model, X, y, cv=Splitter(), groups=['a'] * 12).tolist() == [0.75]
        assert calls[-1][0] == [[0., 100.], [1., 101.]]
    ''')
    result = subprocess.run([sys.executable, '-c', program], capture_output=True, text=True,
                            env=os.environ.copy())
    assert result.returncode == 0, result.stdout + result.stderr


# ---------------------------------------------------------------------------
# LINK 3, THE ROW ORDER (lane/data-ordering-determinism, 2026-09-16).
#
# The fold assignment does not read X. These tests pin the consequence, which
# is a gap and not a bug: an index hash cannot see a permutation, and the
# descriptor can. Both directions are asserted, so neither can quietly stop
# being true.
# ---------------------------------------------------------------------------

class _Clf:
    """The estimator `split_descriptor` reads. Nothing is fitted."""
    _estimator_type = 'classifier'

    def get_params(self, deep=False):
        return {}


def _ordered_fixture(n=48):
    """Rows whose VALUES differ everywhere and whose labels alternate.

    Not uniform, deliberately: uniform test data hides a permutation, and a
    fixture that hides one would make every assertion below vacuous.
    """
    from mojolearn import Array
    labels = [i % 2 for i in range(n)]
    rows = [[float(i), float(-i), float(i * i % 7) + 0.5] for i in range(n)]
    return Array.from_list(rows, '<f4'), labels


def _within_class_rotation(labels):
    """A permutation that rotates the rows of each class by one, so the label
    SEQUENCE is unchanged and only the rows move."""
    order = list(range(len(labels)))
    for value in sorted(set(labels)):
        rows = [i for i, label in enumerate(labels) if label == value]
        for position, row in enumerate(rows):
            order[row] = rows[(position + 1) % len(rows)]
    assert [labels[i] for i in order] == labels
    return order


@pytest.mark.parametrize('splits', [3, 4])
@pytest.mark.parametrize('classifier', [True, False])
def test_fold_indices_cannot_see_a_label_preserving_permutation(splits, classifier):
    # THE GAP ITSELF. `_default_folds` never reads X, so the indices it yields
    # are identical for the permuted rows while the estimator is fitted on
    # different data. If this ever starts failing, the folds began reading X
    # and `split_descriptor`'s docstring is wrong.
    _, labels = _ordered_fixture()
    order = _within_class_rotation(labels)
    assert order != list(range(len(labels)))
    permuted = [labels[i] for i in order]
    assert list(_default_folds(labels, splits, classifier)) == \
        list(_default_folds(permuted, splits, classifier))


@pytest.mark.parametrize('splits', [3, 4])
def test_split_descriptor_moves_where_the_fold_hash_cannot(splits):
    from mojolearn import Array
    from mojolearn.model_selection import split_descriptor

    X, labels = _ordered_fixture()
    order = _within_class_rotation(labels)
    rows = X.tolist()
    moved = Array.from_list([rows[i] for i in order], '<f4')

    before = split_descriptor(X, labels, estimator=_Clf(), cv=splits)
    after = split_descriptor(moved, [labels[i] for i in order], estimator=_Clf(), cv=splits)

    # The two halves of the finding, asserted against each other.
    assert before['fold_assignment_sha256'] == after['fold_assignment_sha256']
    assert before['y_sha256'] == after['y_sha256']
    assert before['X_sha256'] != after['X_sha256']
    assert before['sha256'] != after['sha256']
    assert before['n_rows'] == after['n_rows'] == len(labels)


def test_split_descriptor_is_reproducible_and_refuses_to_guess_the_splitter():
    from mojolearn.model_selection import SPLIT_DESCRIPTOR_SCHEMA, split_descriptor

    X, labels = _ordered_fixture()
    first = split_descriptor(X, labels, estimator=_Clf(), cv=4)
    assert first == split_descriptor(X, labels, estimator=_Clf(), cv=4)
    assert first['schema'] == SPLIT_DESCRIPTOR_SCHEMA
    # Stratified folds and KFold folds are different splits; describing one as
    # the other would be a descriptor of a run that never happened, which is
    # why the estimator is refused rather than defaulted below.
    explicit = split_descriptor(X, labels, cv=[([0, 1, 2], [3, 4])])
    assert explicit['n_folds'] == 1 and explicit['fold_sizes'] == [[3, 2]]
    assert explicit['X_sha256'] == first['X_sha256']
    assert explicit['sha256'] != first['sha256']
    with pytest.raises(ValueError, match='estimator is required'):
        split_descriptor(X, labels, cv=4)
    with pytest.raises(ValueError, match='estimator is required'):
        split_descriptor(X, labels)


def test_fold_order_sabotage_is_dormant_and_fires_only_with_both_switches(monkeypatch):
    # A DORMANT CONTROL THAT IS NEVER CALLED IS NOT A CONTROL. This lane found
    # the switch defined and unreachable; the arm below is what would have
    # caught that, so it asserts the inert state fails.
    _, labels = _ordered_fixture()
    clean = list(_default_folds(labels, 4, True))

    for environment in ({'MOJOLEARN_FOLD_ORDER_SABOTAGE': '1'},
                        {'MOJOLEARN_HOST_ALLOW_SABOTAGE': '1'}):
        with monkeypatch.context() as patch:
            for name, value in environment.items():
                patch.setenv(name, value)
            assert list(_default_folds(labels, 4, True)) == clean, \
                'one switch alone must leave the folds alone'

    with monkeypatch.context() as patch:
        patch.setenv('MOJOLEARN_FOLD_ORDER_SABOTAGE', '1')
        patch.setenv('MOJOLEARN_HOST_ALLOW_SABOTAGE', '1')
        rotated = list(_default_folds(labels, 4, True))

    assert rotated != clean
    # Every invariant a partition check could test still holds, which is the
    # point of this sabotage: it is a DIFFERENT partition, not a broken one.
    assert [len(test) for _, test in rotated] == [len(test) for _, test in clean]
    held = sorted(row for _, test in rotated for row in test)
    assert held == list(range(len(labels)))
    for train, test in rotated:
        assert not set(train) & set(test)
        assert sorted(train + test) == list(range(len(labels)))
