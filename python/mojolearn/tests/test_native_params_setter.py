"""Public setter probes for the consolidated installed native build."""
import array
import weakref

import pytest

from mojolearn import _expansion_linear as linear
from mojolearn import _mode
from mojolearn import model_selection
from mojolearn._native_params_setter import NativeParamsSetter, _CLASSES, _FITTED_FIELDS
from mojolearn._verify_native_runtime import audit_native_call


def clean(operation, *args, expected_error=None, **kwargs):
    audit = audit_native_call(operation, *args, **kwargs)
    report = audit.report()
    assert not any(report[key] for key in ('python_frames', 'forbidden_native_calls',
        'unresolved_native_calls', 'wrapper_work', 'unobserved_existing_threads')), report
    if expected_error is not None:
        assert isinstance(audit.exception, expected_error), report
        return audit.exception
    return audit.assert_native()


@pytest.mark.parametrize('name', _CLASSES)
def test_public_setter_retains_identity_configuration_and_unrelated_fields(name):
    model = getattr(linear, name)(numeric_mode='identical')
    expected = model.get_params(deep=False)
    user_value = array.array('f', [1.0, 2.0])
    model.__dict__['user_metadata'] = user_value
    model.__dict__['user_note_'] = user_value
    for field in _FITTED_FIELDS:
        model.__dict__[field] = True if field.startswith('_mojolearn_') else array.array('f', [3.0])
    result = clean(model.set_params, numeric_mode='identical')
    assert result is model
    actual = clean(model.get_params, False)
    assert actual.keys() == expected.keys()
    assert all(actual[key] is value for key, value in expected.items())
    assert model.__dict__['user_metadata'] is model.__dict__['user_note_'] is user_value
    assert not any(field in model.__dict__ for field in _FITTED_FIELDS if field != '_mojolearn_params_reset')
    assert not clean(_mode.is_fitted, model)
    model.__dict__['_mojolearn_fitted'] = True
    assert clean(_mode.is_fitted, model)


def test_empty_update_is_an_exact_noop():
    model = linear.SGDRegressor(numeric_mode='identical')
    model.__dict__['coef_'] = array.array('f', [1.0])
    before = dict(model.__dict__)
    assert clean(model.set_params) is model
    assert model.__dict__ == before


def test_nested_stored_constructor_values_keep_child_identity_and_routing():
    # Constructors store parameter objects verbatim. This exercises setter
    # routing before fit validates whether a value is a numerical option.
    child = linear.SGDRegressor(alpha=0.25, numeric_mode='identical')
    model = linear.RidgeCV(alphas=child, numeric_mode='identical')
    value = 0.75
    assert clean(model.set_params, alphas__alpha=value) is model
    assert model.__dict__['alphas'] is child
    assert child.__dict__['alpha'] is value


def test_unknown_root_does_not_mutate_the_parent_or_nested_child():
    child = linear.SGDRegressor(alpha=0.25, numeric_mode='identical')
    model = linear.RidgeCV(alphas=child, numeric_mode='identical')
    before, nested_before = dict(model.__dict__), dict(child.__dict__)
    clean(model.set_params, alphas__alpha=0.75, nonexistent=1, expected_error=ValueError)
    assert model.__dict__ == before and child.__dict__ == nested_before


def test_numeric_mode_capture_and_constructor_validation_timing_are_preserved():
    model = linear.SGDRegressor(numeric_mode='identical')
    marker = 'not-a-supported-mode'
    # These constructors store mode/alpha unchanged; native fit owns value checks.
    assert clean(model.set_params, numeric_mode=marker, alpha=-1.0) is model
    assert model.__dict__['numeric_mode'] is marker
    assert model.__dict__['alpha'] == -1.0
    assert clean(model.set_params, numeric_mode=None) is model
    assert model.__dict__['numeric_mode'] is None


@pytest.mark.parametrize('name', ('RidgeCV', 'LassoCV', 'ElasticNetCV', 'LogisticRegressionCV'))
@pytest.mark.parametrize('repeated', (False, True))
def test_supported_splitter_configuration_can_be_replaced(name, repeated):
    splitter = (model_selection.RepeatedKFold(n_splits=2, n_repeats=2)
                if repeated else model_selection.KFold(n_splits=2))
    model = getattr(linear, name)(cv=splitter, numeric_mode='identical')
    assert clean(model.set_params, cv=3) is model
    assert model.__dict__['cv'] == 3


def test_splitters_without_set_params_keep_the_existing_nested_error():
    splitter = model_selection.KFold(n_splits=2)
    model = linear.RidgeCV(cv=splitter, numeric_mode='identical')
    clean(model.set_params, cv__n_splits=3, expected_error=AttributeError)
    assert model.__dict__['cv'] is splitter and splitter.__dict__['n_splits'] == 2


def test_nested_python_callbacks_are_refused_without_running_hooks():
    calls = []
    class Callback:
        def set_params(self, **kwargs):
            calls.append('setter')
        def __getattribute__(self, name):
            calls.append('lookup')
            return object.__getattribute__(self, name)
    callback = Callback()
    model = linear.RidgeCV(alphas=callback, numeric_mode='identical')
    clean(model.set_params, alphas__value=1, expected_error=TypeError)
    assert calls == []


def test_modified_constructor_and_instance_protocols_are_not_invoked(monkeypatch):
    model = linear.SGDRegressor(numeric_mode='identical')
    calls = []
    def callback(*args, **kwargs):
        calls.append('callback')
    with monkeypatch.context() as patch:
        patch.setattr(linear.SGDRegressor, '__init__', callback)
        clean(model.set_params, alpha=0.5, expected_error=TypeError)
    model.__dict__['get_params'] = callback
    clean(NativeParamsSetter.set_params, model, alpha=0.5, expected_error=TypeError)
    assert not calls


def test_removed_state_cannot_run_python_finalizers():
    calls = []
    class Finalizer:
        def __del__(self):
            calls.append('finalized')
    held = Finalizer()
    model = linear.SGDRegressor(numeric_mode='identical')
    model.__dict__['coef_'] = held
    clean(model.set_params, alpha=0.5, expected_error=TypeError)
    assert model.__dict__['coef_'] is held and not calls


def test_release_admission_does_not_hash_a_python_metaclass():
    calls = []
    class Meta(type):
        def __hash__(cls):
            calls.append('hash')
            return 1
    class Owner(metaclass=Meta):
        pass
    held = Owner()
    model = linear.SGDRegressor(numeric_mode='identical')
    model.__dict__['coef_'] = held
    clean(model.set_params, alpha=0.5, expected_error=TypeError)
    assert model.__dict__['coef_'] is held and calls == []


def test_removed_state_cannot_run_weak_reference_callbacks():
    calls = []
    held = array.array('f', [1.0])
    reference = weakref.ref(held, lambda value: calls.append('weakref'))
    model = linear.SGDRegressor(numeric_mode='identical')
    model.__dict__['coef_'] = held
    clean(model.set_params, alpha=0.5, expected_error=TypeError)
    assert reference() is held and model.__dict__['coef_'] is held and not calls


def test_removed_object_array_cannot_release_a_python_finalizer():
    np = pytest.importorskip('numpy')
    calls = []
    class Finalizer:
        def __del__(self):
            calls.append('finalized')
    held = Finalizer()
    value = np.empty(1, dtype=object)
    value[0] = held
    model = linear.RidgeCV(alphas=value, numeric_mode='identical')
    clean(model.set_params, alphas=(0.1, 1.0), expected_error=TypeError)
    assert model.__dict__['alphas'] is value and value[0] is held and not calls


def test_replaced_container_parameters_do_not_call_hash_hooks_or_recurse_in_python():
    cycle = []
    cycle.append(cycle)
    model = linear.RidgeCV(alphas=cycle, numeric_mode='identical')
    replacement = (0.1, 1.0, 10.0)
    assert clean(model.set_params, alphas=replacement) is model
    assert model.__dict__['alphas'] is replacement


def test_real_fit_then_set_params_requires_refit():
    model = linear.SGDRegressor(max_iter=2, tol=None, numeric_mode='identical')
    x, y = [[0.0, 1.0], [1.0, 2.0], [2.0, 4.0], [3.0, 5.0]], [1.0, 2.0, 3.0, 4.0]
    model.fit(x, y)
    assert clean(_mode.is_fitted, model)
    assert clean(model.set_params, alpha=0.5) is model
    assert not clean(_mode.is_fitted, model)
    assert 'coef_' not in model.__dict__ and 'n_features_in_' not in model.__dict__
    model.fit(x, y)
    assert clean(_mode.is_fitted, model)
