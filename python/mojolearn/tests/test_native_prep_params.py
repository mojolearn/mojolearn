"""Prep getter bodies against the installed native metadata owner.

Method binding happens before these probes. test_native_prep_attributes
separately probes the connected public attribute lookup boundary.
"""
import array

import pytest

from mojolearn import _expansion_prep as prep
from mojolearn._verify_native_runtime import audit_native_call

CLASSES = (
    'BernoulliNB', 'Binarizer', 'CategoricalNB', 'ComplementNB', 'GaussianNB',
    'IterativeImputer', 'KBinsDiscretizer', 'LabelBinarizer', 'LabelEncoder',
    'MultiLabelBinarizer', 'MultinomialNB', 'Normalizer', 'OneHotEncoder',
    'OrdinalEncoder', 'PolynomialFeatures', 'PowerTransformer',
    'QuadraticDiscriminantAnalysis', 'QuantileTransformer', 'RFE', 'SelectKBest',
    'SimpleImputer', 'SplineTransformer', 'TargetEncoder', 'VarianceThreshold',
)


def clean(operation, *args, expected_error=None, **kwargs):
    audit = audit_native_call(operation, *args, **kwargs)
    report = audit.report()
    assert not any(report[key] for key in ('python_frames', 'forbidden_native_calls',
        'unresolved_native_calls', 'wrapper_work', 'unobserved_existing_threads')), report
    if expected_error is not None:
        assert isinstance(audit.exception, expected_error), report
        return audit.exception
    return audit.assert_native()


@pytest.mark.parametrize('name', CLASSES)
def test_declared_parameters_preserve_order_references_and_state(name):
    cls = getattr(prep, name)
    model = cls(estimator=object()) if name == 'RFE' else cls()
    names = cls._parameters
    expected = {key: object.__getattribute__(model, key) for key in names}
    model.__dict__['user_metadata'] = array.array('f', [1.0, 2.0])
    model.__dict__['example_fitted_'] = array.array('f', [3.0])
    before = dict(model.__dict__)
    for deep in (True, False):
        actual = clean(model.get_params, deep=deep)
        assert type(actual) is dict and list(actual) == list(names)
        assert all(actual[key] is value for key, value in expected.items())
    assert model.__dict__.keys() == before.keys()
    assert all(model.__dict__[key] is value for key, value in before.items())


def test_nested_parameter_values_and_ignored_deep_do_not_call_python():
    calls = []
    class Callback:
        def get_params(self, deep=True):
            calls.append('get_params')
        def __getattribute__(self, name):
            calls.append('lookup')
            return object.__getattribute__(self, name)
    class Deep:
        def __bool__(self):
            calls.append('bool')
            return True
    value = Callback()
    model = prep.IterativeImputer(estimator=value)
    result = clean(model.get_params, deep=Deep())
    assert result['estimator'] is value and not calls
    assert not any('__' in name for name in result)


def test_array_and_callable_parameters_are_returned_without_conversion_or_calls():
    calls = []
    class Callback:
        def __call__(self, *args, **kwargs):
            calls.append('call')
        def __array__(self, *args, **kwargs):
            calls.append('array')
    value = Callback()
    model = prep.SelectKBest(score_func=value)
    assert clean(model.get_params)['score_func'] is value
    prior = array.array('f', [0.25, 0.75])
    model = prep.GaussianNB(priors=prior)
    assert clean(model.get_params)['priors'] is prior and not calls


def test_missing_constructor_field_preserves_attribute_error():
    model = prep.Normalizer()
    del model.__dict__['norm']
    clean(model.get_params, expected_error=AttributeError)


def test_passive_subclass_retains_the_builtin_parameter_contract():
    class Passive(prep.Normalizer):
        pass
    model = Passive(norm='l1')
    assert clean(model.get_params) == {'norm': 'l1', 'copy': True}


def test_changed_parameter_property_is_refused_before_callback(monkeypatch):
    calls = []
    model = prep.Normalizer()
    with monkeypatch.context() as patch:
        patch.setattr(prep.Normalizer, 'norm', property(lambda self: calls.append('property')), raising=False)
        clean(model.get_params, expected_error=TypeError)
    assert not calls


def test_python_getter_override_is_refused_before_callback():
    calls = []
    class Callback(prep.Normalizer):
        def __getattribute__(self, name):
            calls.append(name)
            return object.__getattribute__(self, name)
    # Test setup bypasses the application getter; the inherited constructor
    # stores ordinary fields and the native operation must never call the hook.
    model = object.__new__(Callback)
    object.__setattr__(model, 'norm', 'l2')
    object.__setattr__(model, 'copy', True)
    clean(prep._PrepBase.get_params, model, expected_error=TypeError)
    assert not calls
