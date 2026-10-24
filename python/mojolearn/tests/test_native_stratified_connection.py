# SPDX-License-Identifier: Apache-2.0
"""First-seen stratification keeps native helper ownership and repeat order."""
import numpy as np
import pytest

from mojolearn import LinearDiscriminantAnalysis, model_selection as ms
from mojolearn import _backend, _expansion_metrics
from mojolearn._verify_native_runtime import audit_native_call


def _repeat_seed(seed, repeat):
    mask = (1 << 64) - 1
    value = (seed * 0x9E3779B97F4A7C15 + (repeat + 1) * 0x632BE59BD9B4E019) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) >> 1


def _poison_loaders(monkeypatch):
    calls = []
    def forbidden(*args, **kwargs):
        calls.append('loader')
        raise AssertionError('stratification called a Python binding loader')
    monkeypatch.setitem(ms._NATIVE_CONTEXT, 'family_msel_xm', forbidden)
    monkeypatch.setattr(ms, '_x_metrics_binding', forbidden)
    monkeypatch.setattr(_expansion_metrics, '_binding', forbidden)
    monkeypatch.setattr(_backend, 'binding', forbidden)
    return calls


@pytest.mark.parametrize('seed', [13, 29])
def test_repeated_stratified_uses_connected_entry_and_exact_repeat_seed_order(seed, monkeypatch):
    x = np.arange(108, dtype='f4').reshape(27, 4)
    # First-seen label order deliberately differs from sorted label order.
    y = np.asarray([9, 2, 5] * 9, dtype='i4')
    expected = []
    for repeat in range(3):
        expected.extend((np.asarray(tr).copy(), np.asarray(te).copy()) for tr, te in
                        ms.StratifiedKFold(3, shuffle=True, random_state=_repeat_seed(seed, repeat)).split(x, y))
    splitter = ms.RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=seed)
    calls = _poison_loaders(monkeypatch)
    iterator = audit_native_call(splitter.split, x, y).assert_native()
    actual = audit_native_call(list, iterator).assert_native()
    assert len(actual) == 9 and not calls
    for (train, test), (want_train, want_test) in zip(actual, expected):
        np.testing.assert_array_equal(train, want_train)
        np.testing.assert_array_equal(test, want_test)
        assert not set(np.asarray(train)) & set(np.asarray(test))
        assert set(y[np.asarray(test)]) == {2, 5, 9}


def test_permutation_stratification_never_reenters_metrics_loader(monkeypatch):
    labels = np.arange(42, dtype='i4') % 3
    source = np.random.default_rng(43).normal(size=(42, 5)).astype('f4')
    source[:, 0] += labels * 2.5
    source[:, 1] -= labels * 1.5
    model = LinearDiscriminantAnalysis()
    # Retain complete expected values before deliberately replacing loaders.
    expected = ms.permutation_test_score(model, source, labels, cv=3,
                                         n_permutations=3, random_state=17, scoring='accuracy')
    calls = _poison_loaders(monkeypatch)
    actual = audit_native_call(ms.permutation_test_score, model, source, labels, cv=3,
                              n_permutations=3, random_state=17, scoring='accuracy').assert_native()
    assert actual[0] == expected[0] and actual[2] == expected[2]
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2] == (1 + np.count_nonzero(np.asarray(actual[1]) >= actual[0])) / 4
    assert not calls


def test_first_seen_helper_replacement_refuses_without_calling_python(monkeypatch):
    calls = []
    def forbidden(*args):
        calls.append('helper')
        raise AssertionError('Python first-seen helper ran')
    monkeypatch.setitem(ms._NATIVE_CONTEXT['family_msel_entries'], 'x_metrics_first_rows', forbidden)
    x = np.arange(48, dtype='f4').reshape(12, 4)
    y = np.arange(12, dtype='i4') % 2
    observed = audit_native_call(ms.StratifiedKFold(3).split, x, y)
    report = observed.report()
    assert not any(report[key] for key in ('python_frames', 'forbidden_native_calls', 'unresolved_native_calls',
                                           'wrapper_work', 'unobserved_existing_threads')), report
    assert isinstance(observed.exception, TypeError), report
    assert 'compiled helper' in str(observed.exception)
    assert not calls
