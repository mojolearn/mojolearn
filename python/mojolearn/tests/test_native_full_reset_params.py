"""Actual public setter probes; run only against the integrated native build."""
import array
import inspect
import weakref

import pytest

import mojolearn as ml
from mojolearn._native_full_reset_params import _CLASSES, _CONTEXT, NativeFullResetParams
from mojolearn._verify_native_runtime import audit_native_call


def clean(operation, *args, error=None, **kwargs):
    audit = audit_native_call(operation, *args, **kwargs)
    report = audit.report()
    assert not any(report[key] for key in ('python_frames', 'forbidden_native_calls',
        'unresolved_native_calls', 'wrapper_work', 'unobserved_existing_threads')), report
    if error is not None:
        assert isinstance(audit.exception, error), report
        return audit.exception
    return audit.assert_native()


def assert_fields(actual, expected):
    assert tuple(vars(actual)) == tuple(vars(expected))
    for name, value in vars(expected).items():
        # Constants in the original constructor have semantic equality;
        # all caller-owned parameter references must be identical.
        current = vars(actual)[name]
        if type(value) in (str, int, float, bool, type(None)):
            assert current == value
        else:
            assert current is value


def test_kernel_ridge_pilot_full_reset_preserves_dictionary_and_borrowed_parameters():
    alpha = array.array('f', [2.0])
    model = ml.KernelRidge(alpha=alpha, numeric_mode='identical')
    fields = vars(model)
    fields.update(coef_=array.array('f', [1.0]), note=object(), note_=object())
    expected = ml.KernelRidge(alpha=alpha, gamma=0.5, numeric_mode='identical')
    assert clean(model.set_params, gamma=0.5) is model
    assert vars(model) is fields and vars(model)['alpha'] is alpha
    assert_fields(model, expected)


@pytest.mark.parametrize('cls', _CLASSES, ids=lambda cls: cls.__name__)
def test_every_public_full_reset_matches_original_constructor_order_and_reference_identity(cls):
    # The public constructor (itself native since the full-reset constructor
    # lane; its own oracle tests are test_native_full_reset_constructors.py)
    # is outside the native operation audit; set_params never calls it.
    values = {name: object() for name in inspect.signature(cls).parameters}
    model = cls(**values)
    key = next(iter(values))
    replacement = object()
    expected = cls(**(values | {key: replacement}))
    fields = vars(model)
    fields.update(fitted_=array.array('f', [1.0]), caller_note=object())
    assert clean(model.set_params, **{key: replacement}) is model
    assert vars(model) is fields
    assert_fields(model, expected)
    assert clean(model.get_params, False)[key] is replacement


@pytest.mark.parametrize('cls', _CLASSES, ids=lambda cls: cls.__name__)
def test_empty_update_is_noop_even_before_initialization(cls):
    model = object.__new__(cls)
    fields = vars(model)
    fields['sentinel'] = object()
    before = dict(fields)
    assert clean(model.set_params) is model
    assert vars(model) is fields and fields == before


@pytest.mark.parametrize('cls', _CLASSES, ids=lambda cls: cls.__name__)
def test_invalid_root_preserves_exact_error_and_all_state(cls):
    values = {name: object() for name in inspect.signature(cls).parameters}
    model = cls(**values)
    before = dict(vars(model))
    error = clean(model.set_params, unknown=1, error=ValueError)
    assert str(error) == f'Invalid parameter {"unknown"!r} for estimator {cls.__name__}. Valid parameters are: {sorted(values)!r}.'
    assert vars(model) == before


@pytest.mark.parametrize('name', ['KNeighborsClassifier', 'KNeighborsRegressor', 'Lasso', 'StackingRegressor'])
def test_fixed_extra_fields_and_repeated_constructor_writes(name):
    cls = getattr(ml, name)
    signature = inspect.signature(cls)
    required = {key: () for key, value in signature.parameters.items() if value.default is value.empty}
    model = cls(**required)
    values = model.get_params(deep=False)
    expected = cls(**values)
    vars(model)['arbitrary_old_state'] = 8
    assert clean(model.set_params, **{next(iter(values)): next(iter(values.values()))}) is model
    assert_fields(model, expected)
    if name == 'StackingRegressor':
        assert vars(model)['stack_method'] == 'auto'


def test_default_numeric_mode_uses_fresh_constructor_not_old_instance_mode():
    model = ml.KernelRidge(numeric_mode='identical')
    clean(model.set_params, numeric_mode=None)
    assert_fields(model, ml.KernelRidge(numeric_mode=None))


def test_passive_subclass_and_class_numeric_mode_preserve_original_none_rule():
    class Passive(ml.KernelRidge):
        numeric_mode = 'identical'
    model = Passive(alpha=1.0)
    expected = Passive(alpha=2.0, numeric_mode=None)
    clean(model.set_params, alpha=2.0, numeric_mode=None)
    assert_fields(model, expected)
    assert 'numeric_mode' not in vars(model)
    assert model.numeric_mode == 'identical'


@pytest.mark.parametrize('child', [ml.KernelRidge, ml.SGDRegressor, ml.StandardScaler,
                                  ml.Ridge, ml.LogisticRegression, ml.LinearDiscriminantAnalysis])
def test_nested_builtin_keeps_child_identity_and_uses_its_native_setter(child):
    model_child = child()
    key = next(iter(model_child.get_params(deep=False)))
    value = model_child.get_params(deep=False)[key]
    model = ml.BaggingRegressor(estimator=model_child)
    clean(model.set_params, **{'estimator__' + key: value})
    assert vars(model)['estimator'] is model_child


def test_nested_declared_shallow_setter_keeps_its_fitted_state():
    child = ml.SimpleImputer()
    vars(child)['statistics_'] = array.array('f', [1.0])
    marker = vars(child)['statistics_']
    model = ml.BaggingRegressor(estimator=child)
    clean(model.set_params, estimator__strategy='median')
    assert vars(child)['statistics_'] is marker and vars(child)['strategy'] == 'median'


def test_invalid_parent_root_precedes_any_nested_mutation():
    child = ml.KernelRidge(alpha=1.0)
    model = ml.BaggingRegressor(estimator=child)
    before = dict(vars(model))
    clean(model.set_params, estimator__alpha=2.0, invalid=3, error=ValueError)
    assert vars(child)['alpha'] == 1.0 and vars(model) == before


def test_replacement_root_is_resolved_before_nested_updates_in_any_keyword_order():
    old = ml.KernelRidge(alpha=1.0)
    replacement = ml.KernelRidge(alpha=2.0)
    model = ml.BaggingRegressor(estimator=old)
    clean(model.set_params, estimator__alpha=3.0, estimator=replacement)
    assert vars(model)['estimator'] is replacement
    assert vars(old)['alpha'] == 1.0 and vars(replacement)['alpha'] == 3.0


def test_earlier_nested_success_survives_later_child_failure_without_parent_reset():
    first = ml.KernelRidge(alpha=1.0)
    second = ml.KernelRidge(alpha=2.0)
    model = ml.StackingRegressor(estimators=first, final_estimator=second)
    before = dict(vars(model))
    clean(model.set_params, estimators__alpha=3.0, final_estimator__invalid=4, error=ValueError)
    assert vars(first)['alpha'] == 3.0 and vars(second)['alpha'] == 2.0
    assert vars(model) == before


def test_nested_missing_setter_retains_attribute_error():
    model = ml.BaggingRegressor(estimator=None)
    error = clean(model.set_params, estimator__alpha=2.0, error=AttributeError)
    assert str(error) == "'NoneType' object has no attribute 'set_params'"


def test_python_child_callback_is_refused_without_execution():
    calls = []
    class Child:
        def set_params(self, **params):
            calls.append(params)
    model = ml.BaggingRegressor(estimator=Child())
    clean(model.set_params, estimator__alpha=2.0, error=TypeError)
    assert calls == []


@pytest.mark.parametrize('hook', ['__init__', '__new__', '__setattr__', '__getattribute__', 'set_params'])
def test_changed_subclass_hooks_are_refused_before_python_execution(hook):
    calls = []
    class Changed(ml.KernelRidge):
        pass
    model = Changed()
    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError('callback executed')
    setattr(Changed, hook, forbidden)
    try:
        clean(NativeFullResetParams.set_params, model, alpha=2.0, error=TypeError)
        assert calls == []
    finally:
        delattr(Changed, hook)


@pytest.mark.parametrize('field', ['alpha', 'numeric_mode'])
def test_parameter_descriptors_are_refused_without_invocation(field):
    calls = []
    class Descriptor:
        def __get__(self, obj, cls):
            calls.append('get')
        def __set__(self, obj, value):
            calls.append('set')
    class Changed(ml.KernelRidge):
        pass
    model = Changed()
    setattr(Changed, field, Descriptor())
    try:
        clean(NativeFullResetParams.set_params, model, alpha=2.0, error=TypeError)
        assert calls == []
    finally:
        delattr(Changed, field)


def test_fixed_extra_field_descriptor_is_refused_without_invocation():
    calls = []
    class Changed(ml.KNeighborsRegressor):
        pass
    model = Changed()
    Changed._index = property(lambda self: calls.append('get'), lambda self, value: calls.append('set'))
    try:
        clean(NativeFullResetParams.set_params, model, n_neighbors=2, error=TypeError)
        assert calls == []
    finally:
        del Changed._index


def test_old_transitive_python_finalizer_is_refused_before_clearing_fields():
    calls = []
    class Owner:
        def __del__(self):
            calls.append('del')
    owner = Owner()
    model = ml.KernelRidge()
    vars(model)['fitted_'] = {'nested': [owner]}
    before = dict(vars(model))
    clean(model.set_params, alpha=2.0, error=TypeError)
    assert vars(model) == before and calls == []
    vars(model).pop('fitted_')


def test_old_weakref_callback_is_refused_before_reset():
    calls = []
    fitted = array.array('f', [1.0])
    reference = weakref.ref(fitted, lambda value: calls.append(value))
    model = ml.KernelRidge()
    vars(model)['coef_'] = fitted
    before = dict(vars(model))
    clean(model.set_params, alpha=2.0, error=TypeError)
    assert vars(model) == before and reference() is fitted and calls == []


@pytest.mark.parametrize('cls', [ml.Ridge, ml.StandardScaler, ml.LogisticRegression])
def test_nested_native_entry_identity_does_not_exempt_owned_python_finalizers(cls):
    calls = []
    class Owner:
        def __del__(self):
            calls.append('del')
    child = cls()
    owner = Owner()
    vars(child)['caller_owned'] = owner
    witness = weakref.ref(owner)
    del owner
    model = ml.BaggingRegressor(estimator=child)
    key, value = next(iter(child.get_params(deep=False).items()))
    before = tuple(vars(child))
    # Do not retain the old owner in a snapshot: a missing native admission
    # must expose the actual __del__ callback, not merely an expected refusal.
    clean(model.set_params, **{'estimator__' + key: value}, error=TypeError)
    assert tuple(vars(child)) == before and witness() is not None and calls == []
    vars(child).pop('caller_owned')


def test_nested_lda_preserves_its_distinct_shallow_assignment_contract():
    child = ml.LinearDiscriminantAnalysis()
    owner = object()
    vars(child)['caller_owned'] = owner
    model = ml.BaggingRegressor(estimator=child)
    clean(model.set_params, estimator__tol=0.125)
    assert vars(child)['caller_owned'] is owner and vars(child)['tol'] == 0.125


def test_nested_cycle_uses_interpreter_recursion_guard_without_parent_mutation():
    model = ml.BaggingRegressor()
    vars(model)['estimator'] = model
    before = dict(vars(model))
    clean(model.set_params, **{'estimator__' * 1200 + 'n_estimators': 2}, error=RecursionError)
    assert vars(model) == before


def test_retained_arbitrary_parameter_owner_is_not_released_or_cloned():
    class Parameter:
        pass
    value = Parameter()
    model = ml.KernelRidge(alpha=value)
    clean(model.set_params, gamma=0.5)
    assert vars(model)['alpha'] is value


def test_old19_linear_setter_keeps_its_distinct_user_state_contract():
    model = ml.SGDRegressor()
    marker = object()
    vars(model)['caller_note'] = marker
    clean(model.set_params, alpha=0.25)
    assert vars(model)['caller_note'] is marker


def test_deferred_host_setters_are_explicitly_outside_this_runtime_claim():
    assert len(_CONTEXT['deferred_host_classes']) == 24
    for cls in _CONTEXT['deferred_host_classes']:
        assert cls not in _CONTEXT['plans']
        assert cls.set_params is not NativeFullResetParams.set_params
