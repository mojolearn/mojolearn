# SPDX-License-Identifier: Apache-2.0
"""Classification host contracts and encoded ABI; real device gates are separate."""
import ctypes
import warnings

import numpy as np
import pytest

sklearn = pytest.importorskip('sklearn')
from sklearn import metrics as reference
from sklearn.exceptions import UndefinedMetricWarning
from mojolearn import metrics, _backend

NAMES = ('precision_score', 'recall_score', 'f1_score')


class Binding:
    flags = (0, 0, 0)

    def __init__(self, mode):
        self.mode, self.calls = mode, []

    def metrics_numeric_mode(self):
        return {'fast': 0, 'deterministic': 2, 'identical': 1}[self.mode]

    def arrays(self, a, b, n):
        return [np.ctypeslib.as_array((ctypes.c_int32*n).from_address(p)).copy()
                for p in (a, b)]

    def confusion_matrix(self, a, b, out, params):
        n, k, norm = params
        yt, yp = self.arrays(a, b, n)
        self.calls.append(('confusion_matrix', params, yt, yp))
        keep = (yt >= 0) & (yp >= 0)
        counts = np.zeros((k, k), dtype=np.int64)
        np.add.at(counts, (yt[keep], yp[keep]), 1)
        if norm:
            denom = [None, counts.sum(1, keepdims=True),
                     counts.sum(0, keepdims=True), counts.sum()][norm]
            with np.errstate(divide='ignore', invalid='ignore'):
                counts = np.nan_to_num(counts/denom).astype(np.float32)
        dtype = ctypes.c_float if norm else ctypes.c_int64
        np.ctypeslib.as_array((dtype*(k*k)).from_address(out))[:] = counts.ravel()

    def precision_recall_fscore(self, a, b, out, params):
        n, k, avg, pos, zero, selected = params
        yt, yp = self.arrays(a, b, n)
        self.calls.append(('precision_recall_fscore', params, yt, yp))
        average = [None, 'binary', 'micro', 'macro', 'weighted'][avg]
        # Binary may contain an explicitly absent positive id; independent
        # count formulas avoid sklearn rejecting that internal dummy label.
        if avg == 1:
            tp = np.sum((yt == pos) & (yp == pos))
            actual, predicted = np.sum(yt == pos), np.sum(yp == pos)
            values = [tp/predicted if predicted else zero,
                      tp/actual if actual else zero,
                      2*tp/(actual+predicted) if actual+predicted else zero]
        else:
            values = reference.precision_recall_fscore_support(
                yt, yp, labels=list(range(selected)), average=average,
                zero_division=zero)[:3]
        width = selected if avg == 0 else 1
        result = np.ctypeslib.as_array((ctypes.c_float*(3*width+3)).from_address(out))
        result[:3*width] = np.asarray(values).ravel()
        result[3*width:] = self.flags


@pytest.fixture
def binding(monkeypatch):
    modules = {m: Binding(m) for m in ('fast', 'deterministic', 'identical')}
    monkeypatch.setattr(_backend, 'default_mode', lambda: 'fast')
    monkeypatch.setattr(_backend, 'binding', lambda name, mode=None: modules[mode])
    return modules


@pytest.mark.parametrize('mode', ['fast', 'deterministic', 'identical'])
@pytest.mark.parametrize('normalize', [None, 'true', 'pred', 'all'])
def test_confusion_order_subset_normalize_and_mode(binding, mode, normalize):
    yt = np.array(['a', 'b', 'c', 'b', 'a'])
    yp = np.array(['b', 'b', 'a', 'c', 'a'])
    before = yt.copy()
    result = metrics.confusion_matrix(yt, yp, labels=['b', 'a', 'absent'],
                                      normalize=normalize, numeric_mode=mode)
    expected = reference.confusion_matrix(yt, yp, labels=['b', 'a', 'absent'],
                                          normalize=normalize)
    np.testing.assert_allclose(result, expected, atol=1e-7)
    assert np.asarray(result).dtype == (np.int64 if normalize is None else np.float32)
    assert len(binding[mode].calls) == 1
    np.testing.assert_array_equal(yt, before)
    assert -1 in binding[mode].calls[0][2]


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('average', [None, 'micro', 'macro', 'weighted'])
@pytest.mark.parametrize('zero', [0, 1])
def test_prf_retains_errors_against_excluded_labels(binding, name, average, zero):
    yt, yp = [1, 1, 2, 0, 0], [0, 1, 0, 2, 0]
    options = dict(labels=[2, 0, 9], average=average, zero_division=zero)
    result = getattr(metrics, name)(yt, yp, numeric_mode='identical', **options)
    expected = getattr(reference, name)(yt, yp, **options)
    np.testing.assert_allclose(result, expected, atol=1e-7)
    if average is None:
        assert np.asarray(result).dtype == np.float32 and result.shape == (3,)
    else:
        assert isinstance(result, float)
    params = binding['identical'].calls[0][1]
    assert params[1] == 4 and params[-1] == 3


@pytest.mark.parametrize('name', NAMES)
def test_weighted_zero_support_falls_back_to_unweighted_mean(binding, name):
    options = dict(labels=[0, 2], average='weighted', zero_division=1)
    assert getattr(metrics, name)([1, 1], [0, 1], **options) == getattr(reference, name)([1, 1], [0, 1], **options)


@pytest.mark.parametrize('name', NAMES)
def test_binary_strings_absent_positive_and_large_integers(binding, name):
    assert getattr(metrics, name)(['no', 'yes'], ['yes', 'yes'], pos_label='yes') == pytest.approx(
        getattr(reference, name)(['no', 'yes'], ['yes', 'yes'], pos_label='yes'))
    assert getattr(metrics, name)([0, 0], [0, 0], zero_division=1) == 1
    big = 2**80
    got = getattr(metrics, name)([big, -big], [big, big], pos_label=big)
    assert np.isfinite(got)
    assert set(binding['fast'].calls[-1][2]) == {0, 1}


@pytest.mark.parametrize('name,index', list(zip(NAMES, range(3))))
def test_warning_uses_requested_metric_native_flag(binding, name, index):
    flags = [0, 0, 0]
    flags[index] = 1
    binding['fast'].flags = flags
    with pytest.warns(UndefinedMetricWarning):
        getattr(metrics, name)([0], [0])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        getattr(metrics, name)([0], [0], zero_division=0)
        assert not caught


@pytest.mark.parametrize('name', ('confusion_matrix',) + NAMES)
@pytest.mark.parametrize('true,pred,error', [([], [], ValueError), ([[0]], [[0]], ValueError),
    ([0], [0, 1], ValueError), ([0.], [0.], TypeError), ([0, 'x'], [0, 'x'], TypeError),
    ([None], [None], TypeError), ([0], ['0'], TypeError)])
def test_reject_before_native(binding, name, true, pred, error):
    with pytest.raises(error):
        getattr(metrics, name)(true, pred)
    assert not binding['fast'].calls


@pytest.mark.parametrize('name', ('confusion_matrix',) + NAMES)
def test_unsupported_weights_duplicate_labels_and_mode(binding, name):
    method = getattr(metrics, name)
    extra = {} if name == 'confusion_matrix' else {'average': 'macro'}
    with pytest.raises(NotImplementedError, match='sample_weight'):
        method([0], [0], sample_weight=[1])
    for labels in ([], [0, 0]):
        with pytest.raises(ValueError):
            method([0], [0], labels=labels, **extra)
    with pytest.raises(ValueError, match='numeric_mode'):
        method([0], [0], numeric_mode='typo')
    assert not binding['fast'].calls


@pytest.mark.parametrize('name', NAMES)
def test_prf_options_rejected(binding, name):
    method = getattr(metrics, name)
    for bad in ('samples', [], 1):
        with pytest.raises(ValueError, match='average'):
            method([0], [0], average=bad)
    for bad in (2, float('nan'), 'invalid', True):
        with pytest.raises(ValueError, match='zero_division'):
            method([0], [0], zero_division=bad)
    with pytest.raises(ValueError, match='two observed'):
        method([0, 1, 2], [0, 1, 2])
    with pytest.raises(ValueError, match='pos_label'):
        method([0, 1], [0, 1], pos_label=2)
    assert not binding['fast'].calls


def test_confusion_allocation_bound_and_options(binding):
    with pytest.raises(ValueError, match='4096'):
        metrics.confusion_matrix([0], [0], labels=list(range(4097)))
    for norm in (0, [], 'rows'):
        with pytest.raises(ValueError, match='normalize'):
            metrics.confusion_matrix([0], [0], normalize=norm)
    with pytest.raises(ValueError, match='At least one'):
        metrics.confusion_matrix([0], [1], labels=[1])
    assert not binding['fast'].calls


@pytest.mark.parametrize('name', ('confusion_matrix',) + NAMES)
def test_options_keyword_only_and_exports(binding, name):
    assert name in metrics.__all__
    with pytest.raises(TypeError):
        getattr(metrics, name)([0], [0], 'identical')
    assert not binding['fast'].calls
