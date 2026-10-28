"""Public lifecycle controls for the integrated manifold owner (not yet run)."""
import numpy as np
import pytest

from mojolearn import ClassicalMDS, Isomap, LocallyLinearEmbedding, MDS, TSNE
from mojolearn.exceptions import NotFittedError


CASES = [
    (ClassicalMDS, {}),
    (Isomap, {'n_neighbors': 2}),
    (LocallyLinearEmbedding, {'n_neighbors': 3}),
    (MDS, {'max_iter': 2, 'n_init': 1}),
    (TSNE, {'perplexity': 2, 'max_iter': 2, 'init': 'random'}),
]


@pytest.mark.parametrize('cls, settings', CASES)
def test_public_fit_shells_have_no_inherited_python_envelope(cls, settings):
    for name in ('fit', 'fit_transform'):
        method = cls.__dict__[name]
        assert not hasattr(method, '__wrapped__')
        assert method._mojolearn_native_public_operation
    class Passive(cls):
        pass
    assert Passive.fit is cls.fit
    assert Passive.fit_transform is cls.fit_transform


@pytest.mark.parametrize('cls, settings', CASES)
@pytest.mark.parametrize('x, message', [
    (np.array(1, dtype=np.float32), 'Expected 2D array, got scalar'),
    (np.ones(3, dtype=np.float32), 'Expected 2D array, got 1D'),
    (np.empty((0, 3), dtype=np.float32), '0 sample'),
    (np.empty((3, 0), dtype=np.float32), '0 feature'),
    (np.ones((3, 2), dtype=np.complex64), 'Complex data not supported'),
])
def test_input_refusals_precede_numeric_execution(cls, settings, x, message):
    with pytest.raises(ValueError, match=message):
        cls(**settings).fit(x)


@pytest.mark.parametrize('cls, settings', CASES)
def test_ignored_target_checks_stay_in_the_native_call(cls, settings):
    x = np.ones((6, 4), dtype=np.float32)
    with pytest.raises(ValueError, match='inconsistent numbers of samples'):
        cls(**settings).fit(x, np.zeros(5, dtype=np.float64))
    with pytest.raises(ValueError, match='Input y contains NaN or infinity'):
        cls(**settings).fit(x, np.full(6, np.inf, dtype=np.float64))
    with pytest.raises(ValueError, match='complex y'):
        cls(**settings).fit(x, np.ones(6, dtype=np.complex128))


@pytest.mark.parametrize('cls', [Isomap, LocallyLinearEmbedding])
def test_transform_fitted_guard_precedes_input_protocols(cls):
    calls = []
    class Input:
        def __array__(self, *args, **kwargs):
            calls.append('array')
            raise AssertionError('input protocol ran')
    with pytest.raises(NotFittedError, match='before using transform'):
        cls().transform(Input())
    assert calls == []


def test_native_matrix_owner_avoids_legacy_matrix_hooks(monkeypatch):
    from mojolearn import _expansion_decomp
    calls = []
    original = _expansion_decomp._M.__getattribute__
    def observed(self, name):
        calls.append(name)
        return original(self, name)
    monkeypatch.setattr(_expansion_decomp._M, '__getattribute__', observed)
    x = np.random.default_rng(17).normal(size=(8, 4)).astype(np.float32)
    model = ClassicalMDS(n_components=2)
    result = model.fit_transform(x)
    assert type(result) is np.ndarray and result.shape == (8, 2)
    assert result.dtype == np.float32
    assert vars(model)['_mojolearn_fitted'] is True
    assert type(vars(model)['embedding_m_']).__name__ == '_ManifoldMatrix'
    assert calls == []


def test_after_fit_callback_is_refused_without_running_it():
    calls = []
    model = ClassicalMDS()
    vars(model)['_mojolearn_after_fit'] = lambda: calls.append('after')
    with pytest.raises(TypeError, match='after-fit callbacks'):
        model.fit(np.ones((6, 4), dtype=np.float32))
    assert calls == []
