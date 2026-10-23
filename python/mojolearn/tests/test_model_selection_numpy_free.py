# SPDX-License-Identifier: Apache-2.0
"""Fold metadata matches sklearn; built-in CV never needs its dependencies."""
import os
import subprocess
import sys
import textwrap

import pytest

from mojolearn.tests._fold_reference import default_folds as _default_folds


def _native(call, *args, error=None, **kwargs):
    from mojolearn._verify_native_runtime import audit_native_call
    observed = audit_native_call(call, *args, **kwargs)
    report = observed.report()
    assert not any(report[key] for key in ('python_frames', 'forbidden_native_calls', 'unresolved_native_calls',
                                           'wrapper_work', 'unobserved_existing_threads')), report
    if error is not None:
        assert isinstance(observed.exception, error), report
        return observed.exception
    return observed.assert_native()


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


def test_pipeline_callbacks_refuse_and_native_cv_clones_keep_independent_parameters(monkeypatch):
    pipeline = pytest.importorskip('sklearn.pipeline')
    preprocessing = pytest.importorskip('sklearn.preprocessing')
    from mojolearn import Array, GaussianNB, model_selection as ms
    original = pipeline.Pipeline([('scale', preprocessing.StandardScaler())])
    touched = []
    def forbidden(*args, **kwargs):
        touched.append('external clone/get_params')
        raise AssertionError('Python pipeline protocol executed')
    monkeypatch.setattr(pipeline.Pipeline, '__sklearn_clone__', forbidden)
    monkeypatch.setattr(pipeline.Pipeline, 'get_params', forbidden)
    x = Array.from_list([[float(i % 5) + 2 * (i % 2), float(i // 5)] for i in range(20)], '<f4')
    y = Array.from_list([0, 1] * 10, '<i4')
    _native(ms.cross_val_score, original, x, y, cv=2, error=TypeError)
    assert touched == []
    priors = [.4, .6]
    model = GaussianNB(priors=priors)
    result = _native(ms.cross_validate, model, x, y, cv=2, return_estimator=True, error_score='raise')
    clones = result['estimator']
    assert len(clones) == 2 and clones[0] is not clones[1]
    assert all(type(clone) is GaussianNB and clone is not model for clone in clones)
    assert all(clone.priors == priors and clone.priors is not priors for clone in clones)
    assert clones[0].priors is not clones[1].priors
    clones[0].priors[0] = .1
    assert clones[1].priors == priors == [.4, .6]
    assert not hasattr(model, 'classes_')


def test_default_and_explicit_cv_without_numpy_or_sklearn():
    # Fresh process guarantees an already-imported dependency cannot mask reach.
    program = textwrap.dedent('''
        import builtins, sys
        original_import = builtins.__import__
        def blocked(name, *args, **kwargs):
            if name.split('.')[0] in ('numpy', 'sklearn'):
                raise AssertionError('unexpected dependency: ' + name)
            return original_import(name, *args, **kwargs)
        builtins.__import__ = blocked
        from mojolearn import Array, GaussianNB, model_selection as ms
        from mojolearn._verify_native_runtime import audit_native_call
        calls = []
        def native(call, *args, error=None, **kwargs):
            observed = audit_native_call(call, *args, **kwargs)
            report = observed.report()
            assert not any(report[key] for key in ('python_frames', 'forbidden_native_calls', 'unresolved_native_calls',
                                                    'wrapper_work', 'unobserved_existing_threads')), report
            if error is not None:
                assert isinstance(observed.exception, error), report
                return
            return observed.assert_native()
        rows = [[float(i % 5) + 2 * (i % 2), float(i // 5)] for i in range(20)]
        labels = [0, 1] * 10
        X = Array.from_list(rows, '<f4')
        y = Array.from_list(labels, '<i4')
        priors = [.4, .6]
        model = GaussianNB(priors=priors)
        # Independent unshuffled stratification: two rows of each class
        # per held-out fold, from the alternating label fixture.
        folds = [([i for i in range(20) if not f * 4 <= i < (f + 1) * 4],
                  list(range(f * 4, (f + 1) * 4))) for f in range(5)]
        expected = []
        for train, test in folds:
            fitted = GaussianNB(priors=priors).fit(Array.from_list([rows[i] for i in train], '<f4'),
                                                  Array.from_list([labels[i] for i in train], '<i4'))
            expected.append(fitted.score(Array.from_list([rows[i] for i in test], '<f4'),
                                         Array.from_list([labels[i] for i in test], '<i4')))
        assert native(ms.cross_val_score, model, X, y).tolist() == expected
        assert native(ms.cross_val_score, model, X, y, cv=folds).tolist() == expected
        result = native(ms.cross_validate, model, X, y, cv=folds, return_estimator=True, error_score='raise')
        assert result['test_score'].tolist() == expected
        assert len({id(item) for item in result['estimator']}) == 5
        assert all(item is not model and item.priors is not priors for item in result['estimator'])
        assert model.priors is priors and priors == [.4, .6] and not hasattr(model, 'classes_')
        class Estimator:
            def __getattribute__(self, name):
                calls.append(name)
                raise AssertionError('external estimator hook executed')
        class Splitter:
            def split(self, X, y, groups):
                calls.append('split')
                raise AssertionError('external splitter executed')
        native(ms.cross_val_score, Estimator(), X, y, cv=2, error=TypeError)
        native(ms.cross_val_score, model, X, y, cv=Splitter(), groups=['a'] * 20, error=TypeError)
        assert calls == []
        assert not any(name.split('.')[0] in ('numpy', 'sklearn') for name in sys.modules)
    ''')
    result = subprocess.run([sys.executable, '-c', program], capture_output=True, text=True,
                            env=os.environ.copy(), timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr


# ---------------------------------------------------------------------------
# LINK 3, THE ROW ORDER (lane/data-ordering-determinism, 2026-09-16).
#
# The fold assignment does not read X. These tests pin the consequence, which
# is a gap and not a bug: an index hash cannot see a permutation, and the
# descriptor can. Both directions are asserted, so neither can quietly stop
# being true.
# ---------------------------------------------------------------------------

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
    from mojolearn import Array, GaussianNB
    from mojolearn.model_selection import split_descriptor

    X, labels = _ordered_fixture()
    order = _within_class_rotation(labels)
    rows = X.tolist()
    moved = Array.from_list([rows[i] for i in order], '<f4')

    before = _native(split_descriptor, X, labels, estimator=GaussianNB(), cv=splits)
    after = _native(split_descriptor, moved, [labels[i] for i in order], estimator=GaussianNB(), cv=splits)

    # The two halves of the finding, asserted against each other.
    assert before['fold_assignment_sha256'] == after['fold_assignment_sha256']
    assert before['y_sha256'] == after['y_sha256']
    assert before['X_sha256'] != after['X_sha256']
    assert before['sha256'] != after['sha256']
    assert before['n_rows'] == after['n_rows'] == len(labels)


def test_split_descriptor_is_reproducible_and_refuses_to_guess_the_splitter():
    from mojolearn import GaussianNB
    from mojolearn.model_selection import SPLIT_DESCRIPTOR_SCHEMA, split_descriptor

    X, labels = _ordered_fixture()
    first = _native(split_descriptor, X, labels, estimator=GaussianNB(), cv=4)
    assert first == _native(split_descriptor, X, labels, estimator=GaussianNB(), cv=4)
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
    touched = []
    class ExternalClassifier:
        def __getattribute__(self, name):
            touched.append(name)
            raise AssertionError('external classifier metadata hook ran')
    _native(split_descriptor, X, labels, estimator=ExternalClassifier(), cv=4, error=TypeError)
    assert touched == []


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
