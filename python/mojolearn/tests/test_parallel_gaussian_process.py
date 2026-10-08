# SPDX-License-Identifier: Apache-2.0
"""Canonical class composition tests, independent of native GPU qualification."""
from types import SimpleNamespace
import numpy as np
import pytest
from mojolearn import parallel_gaussian_process as pg
from mojolearn._gpc_impl import GaussianProcessClassifier as GPC
from mojolearn._buffer import Array
from mojolearn._parallel_worker import execute


class Pool:
    closed = False
    calls = []
    fail = False
    def __init__(self, devices):
        type(self).closed = False
    def map(self, requests):
        type(self).calls = requests
        if self.fail:
            raise RuntimeError('class failed')
        return [execute(r) for r in requests]
    def close(self):
        type(self).closed = True


def fit_binary(self, ext, x, y01, *kernel, k=None):
    # With `k` the worker hands the int32 class codes and the binding builds
    # the targets `code == k` (_gpc_impl._fit_binary, lane fam2-kernel-gp).
    if k is not None:
        y01 = Array.from_list([1.0 if int(c) == k else 0.0 for c in y01.tolist()], '<f4')
    bits = y01.tolist()
    return SimpleNamespace(y_train_=y01, L_=x, pi_=y01, W_sr_=y01,
                           n_iter_=3, log_marginal_likelihood_value_=sum(bits) / 3)


def latent(self, ext, est, q, want_proba, out_kind=0):
    # Includes exact ties and zeros; class identity changes remaining rows.
    target = est.y_train_.tolist().index(1.0)
    p = [0.0, 0.5, (target + 1) / 7, 0.0][:q.shape[0]]
    mean = Array.from_list([-0.0, 1.0, -1.0, 0.0][:q.shape[0]], '<f4')
    if not out_kind:
        return mean, None, Array.from_list(p, '<f8')
    # out_kind 1 / 2: the binding's fourth item, a binary model's predict
    # codes or its [1 - p, p] rows (gaussian_process/unnorm.mojo::gpc_binary_out)
    out = (Array.from_list([int(v > 0.5) for v in p], '<i8') if out_kind == 1
           else Array.from_list([[1.0 - v, v] for v in p], '<f8'))
    return mean, None, Array.from_list(p, '<f8'), out


@pytest.fixture(autouse=True)
def setup(monkeypatch):
    Pool.fail = False
    monkeypatch.setattr(pg, 'DevicePool', Pool)
    monkeypatch.setattr('mojolearn._backend.default_mode', lambda: 'identical')
    # The real GP binding stays: the one-vs-rest combine (`gpc_ovr_combine`)
    # is native on every build. Its all-classes doors (`gp_idn_caps` bit 0:
    # gpc_fit_all / gpc_predict_all) are hidden so the plain reference takes
    # the per-class fit and latent the worker shards take, both faked here.
    real_extension = GPC._extension

    class _PerClass:
        def __init__(self, ext):
            self._ext = ext

        def __getattr__(self, name):
            if name == 'gp_idn_caps':
                raise AttributeError(name)
            return getattr(self._ext, name)
    monkeypatch.setattr(GPC, '_extension', lambda self: _PerClass(real_extension(self)))
    monkeypatch.setattr(GPC, '_fit_binary', fit_binary)
    monkeypatch.setattr(GPC, '_latent', latent)
    from mojolearn._cpu_reference import reference_training
    with reference_training():
        yield


@pytest.mark.parametrize('labels', [[9, 3, 9, 3, 9], ['c', 'a', 'b', 'a', 'c']])
def test_fit_targets_state_and_predictions_equal_plain(labels):
    x = np.arange(15, dtype=np.float32).reshape(5, 3)
    expected = GPC().fit(x, labels)
    model = GPC()
    assert pg.fit_gaussian_process_classifier(model, x, labels, devices=(1, 0)) is model
    assert Pool.closed
    assert model.classes_ == expected.classes_
    assert model.n_iter_ == expected.n_iter_
    assert model.log_marginal_likelihood_value_ == expected.log_marginal_likelihood_value_
    assert [e.y_train_.tolist() for e in model.estimators_] == [e.y_train_.tolist() for e in expected.estimators_]
    for method in ('predict', 'predict_proba'):
        actual = pg.predict_gaussian_process_classifier(model, x[:4], method=method, devices=(1, 0))
        wanted = getattr(expected, method)(x[:4])
        assert np.asarray(actual).tobytes() == np.asarray(wanted).tobytes()
        assert np.asarray(actual).shape == np.asarray(wanted).shape
        assert all('estimators_' not in state.__dict__ for _, state, _ in Pool.calls)
        assert Pool.closed


def test_failure_does_not_publish_partial_fit():
    model = GPC().fit(np.ones((3, 2), np.float32), [1, 2, 3])
    before = model.__dict__.copy()
    Pool.fail = True
    with pytest.raises(RuntimeError, match='class failed'):
        pg.fit_gaussian_process_classifier(model, np.ones((3, 2), np.float32), [3, 4, 5])
    assert model.__dict__ == before
    assert Pool.closed


@pytest.mark.parametrize('labels', [[1, 1], [1], []])
def test_invalid_targets(labels):
    with pytest.raises(ValueError):
        pg.fit_gaussian_process_classifier(GPC(), np.ones((2, 3), np.float32), labels)


def test_prediction_validation():
    with pytest.raises(ValueError, match='method'):
        pg.predict_gaussian_process_classifier(GPC(), np.ones((2, 3), np.float32), method='score')
    with pytest.raises(ValueError, match='fit'):
        pg.predict_gaussian_process_classifier(GPC(), np.ones((2, 3), np.float32))
