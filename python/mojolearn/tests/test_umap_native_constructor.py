"""Focused public UMAP constructor/reset controls for the integrated native owner."""
import inspect

import numpy as np
import pytest

from mojolearn import UMAP


def test_constructor_and_reset_are_one_native_call_wrappers():
    assert UMAP.__init__.__module__ == 'mojolearn._umap_impl'
    assert not hasattr(UMAP.__init__, '__wrapped__')
    assert 'numeric_mode' in inspect.signature(UMAP).parameters
    model = UMAP(n_neighbors=np.int64(3), min_dist=np.float32(0.2), numeric_mode='identical')
    assert model.n_neighbors == 3
    assert model.min_dist == np.float32(0.2)
    assert model.numeric_mode == 'identical'


@pytest.mark.parametrize('settings, message', [
    ({'n_neighbors': True}, 'n_neighbors must be an integer'),
    ({'n_neighbors': 1}, 'n_neighbors must be in'),
    ({'n_components': 33}, 'n_components must be in'),
    ({'n_epochs': 0}, 'n_epochs must be in'),
    ({'random_state': -1}, 'random_state must be in'),
    ({'min_dist': float('nan')}, 'min_dist must be a finite float32'),
    ({'min_dist': 2.0}, '0 <= min_dist <= spread'),
    ({'learning_rate': 0}, 'learning_rate must be positive'),
    ({'repulsion_strength': -1}, 'repulsion_strength must be nonnegative'),
    ({'set_op_mix_ratio': 2}, 'set_op_mix_ratio must be in'),
    ({'local_connectivity': -1}, 'local_connectivity must be >= 0'),
    ({'metric': 'minkowski', 'metric_kwds': {'p': 0}}, 'minkowski p must be positive'),
    ({'a': 1}, 'a and b must be given together'),
    ({'a': -1, 'b': 1}, 'a and b must be positive'),
    ({'init': 'bad'}, 'init='),
])
def test_constructor_preserves_parameter_refusals(settings, message):
    with pytest.raises(ValueError, match=message):
        UMAP(**settings)


@pytest.mark.parametrize('field, hook', [('n_neighbors', '__index__'), ('min_dist', '__float__'), ('densmap', '__bool__')])
def test_constructor_refuses_scalar_protocols_before_invoking_them(field, hook):
    calls = []

    def invoked(self):
        calls.append(hook)
        return 3 if hook == '__index__' else (0.2 if hook == '__float__' else False)

    protocol = type('Protocol', (), {hook: invoked})()
    with pytest.raises((TypeError, ValueError)):
        UMAP(**{field: protocol})
    assert calls == []


def test_constructor_refuses_metric_mapping_protocols():
    calls = []

    class Mapping(dict):
        def __iter__(self):
            calls.append('iter')
            return super().__iter__()

        def __bool__(self):
            calls.append('bool')
            return True

    with pytest.raises(TypeError, match='exact native dictionary'):
        UMAP(metric='minkowski', metric_kwds=Mapping(p=2))
    assert calls == []


def test_failed_set_params_preserves_previous_dictionary_and_fitted_state():
    model = UMAP(n_neighbors=3, numeric_mode='identical')
    state = vars(model)
    marker = np.arange(6, dtype=np.float32).reshape(3, 2)
    state['embedding_'] = marker
    before = dict(state)
    with pytest.raises(ValueError, match='n_neighbors must be in'):
        model.set_params(n_neighbors=1)
    assert vars(model) is state
    assert list(state) == list(before)
    assert all(state[key] is value for key, value in before.items())
    model.set_params(n_neighbors=4)
    assert model.n_neighbors == 4
    assert 'embedding_' not in vars(model)


def test_reinitialization_validation_precedes_mode_injection():
    model = UMAP(n_neighbors=3, numeric_mode='identical')
    state = vars(model)
    with pytest.raises(ValueError, match='n_neighbors must be in'):
        model.__init__(n_neighbors=1, numeric_mode='fast')
    assert vars(model) is state
    assert state['n_neighbors'] == 1
    assert state['numeric_mode'] == 'identical'


def test_constructor_preserves_unsupported_objective_error_classes():
    with pytest.raises(NotImplementedError, match='densmap=True'):
        UMAP(densmap=True)
    with pytest.raises(NotImplementedError, match='output_metric='):
        UMAP(output_metric='cosine')


def test_zero_dimensional_native_arrays_remain_supported_constructor_scalars():
    model = UMAP(n_neighbors=np.array(3, dtype=np.int64), min_dist=np.array(0.2, dtype='>f8'))
    assert model.n_neighbors == 3
    assert model.min_dist == 0.2
