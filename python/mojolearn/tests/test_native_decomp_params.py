"""Public decomposition getter bodies against the installed native owner.

Method lookup and constructors are test setup; these probes qualify the getter
operation, not every method or the earlier public attribute lookup boundary.
"""
import array

import pytest

from mojolearn import _expansion_decomp as decomp
from mojolearn._verify_native_runtime import audit_native_call

CLASSES = (
    'AlternatingLeastSquares', 'CCA', 'ClassicalMDS', 'DictionaryLearning',
    'EllipticEnvelope', 'FactorAnalysis', 'FastICA', 'GaussianRandomProjection',
    'IncrementalPCA', 'Isomap', 'LatentDirichletAllocation',
    'LocallyLinearEmbedding', 'MDS', 'MinCovDet', 'MiniBatchDictionaryLearning',
    'MiniBatchSparsePCA', 'NMF', 'PLSCanonical', 'PLSRegression', 'SparseCoder',
    'SparsePCA', 'SparseRandomProjection',
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
    cls = getattr(decomp, name)
    model = cls(dictionary=array.array('f', [1.0, 0.0])) if name == 'SparseCoder' else cls()
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


def test_deep_is_ignored_and_nested_parameter_callbacks_do_not_run():
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
    model = decomp.ClassicalMDS(metric_params=value)
    result = clean(model.get_params, deep=Deep())
    assert result['metric_params'] is value and not calls
    assert not any('__' in name for name in result)


def test_buffer_parameter_is_returned_without_conversion():
    dictionary = array.array('f', [1.0, 0.0, 0.0, 1.0])
    model = decomp.SparseCoder(dictionary=dictionary)
    assert clean(model.get_params)['dictionary'] is dictionary


def test_callable_parameter_is_borrowed_without_execution():
    calls = []

    class Callback:
        def __call__(self, *args, **kwargs):
            calls.append('call')

        def __array__(self, *args, **kwargs):
            calls.append('array')

    callback = Callback()
    model = decomp.DictionaryLearning(callback=callback)
    assert clean(model.get_params)['callback'] is callback and not calls


def test_missing_constructor_field_preserves_attribute_error():
    model = decomp.ClassicalMDS()
    del model.__dict__['metric']
    clean(model.get_params, expected_error=AttributeError)


def test_passive_subclass_retains_builtin_parameter_contract():
    class Passive(decomp.ClassicalMDS):
        pass

    model = Passive(n_components=3)
    expected = {name: object.__getattribute__(model, name) for name in model._parameters}
    actual = clean(model.get_params)
    assert list(actual) == list(expected)
    assert all(actual[name] is value for name, value in expected.items())


def test_changed_parameter_property_is_refused_before_callback(monkeypatch):
    calls = []
    model = decomp.ClassicalMDS()
    operation = model.get_params
    with monkeypatch.context() as patch:
        patch.setattr(decomp.ClassicalMDS, 'metric', property(lambda self: calls.append('property')), raising=False)
        clean(operation, expected_error=TypeError)
    assert not calls


def test_python_getter_override_is_refused_before_callback():
    calls = []

    class Callback(decomp.ClassicalMDS):
        def __getattribute__(self, name):
            calls.append(name)
            return object.__getattribute__(self, name)

    model = object.__new__(Callback)
    object.__setattr__(model, 'n_components', 2)
    object.__setattr__(model, 'metric', 'euclidean')
    object.__setattr__(model, 'metric_params', None)
    object.__setattr__(model, 'numeric_mode', None)
    clean(decomp._Base.get_params, model, expected_error=TypeError)
    assert not calls


def test_each_call_returns_an_independent_dictionary():
    model = decomp.ClassicalMDS()
    first = clean(model.get_params)
    metric = first['metric']
    first['metric'] = object()
    first['extra'] = object()
    second = clean(model.get_params)
    assert second is not first and second['metric'] is metric and 'extra' not in second
