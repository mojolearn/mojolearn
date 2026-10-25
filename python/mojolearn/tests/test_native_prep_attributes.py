# SPDX-License-Identifier: Apache-2.0
"""Installed prep attribute slots; constructors and method bodies are separate gates."""
import gc
import weakref

import pytest

from mojolearn import _expansion_prep as prep
from mojolearn import _native_public_attributes as owner
from mojolearn._array import Array
from mojolearn._verify_native_runtime import audit_native_call

CLASSES = (
    'BernoulliNB', 'Binarizer', 'CategoricalNB', 'ComplementNB', 'GaussianNB',
    'IterativeImputer', 'KBinsDiscretizer', 'LabelBinarizer', 'LabelEncoder',
    'MultiLabelBinarizer', 'MultinomialNB', 'Normalizer', 'OneHotEncoder',
    'OrdinalEncoder', 'PolynomialFeatures', 'PowerTransformer',
    'QuadraticDiscriminantAnalysis', 'QuantileTransformer', 'RFE', 'SelectKBest',
    'SimpleImputer', 'SplineTransformer', 'TargetEncoder', 'VarianceThreshold',
)
CLASSIFIERS = ('BernoulliNB', 'CategoricalNB', 'ComplementNB', 'GaussianNB',
               'MultinomialNB', 'QuadraticDiscriminantAnalysis')


def clean(operation, *args, error=None):
    audit = audit_native_call(operation, *args)
    report = audit.report()
    assert not any(report[key] for key in ('python_frames', 'forbidden_native_calls',
        'unresolved_native_calls', 'wrapper_work', 'unobserved_existing_threads')), report
    if error is not None:
        assert isinstance(audit.exception, error), report
        return audit.exception
    return audit.assert_native()


def read(model, name):
    return getattr(model, name)


def public_reader(**markers):
    namespace = {'__name__': 'prep_attribute_application', **markers}
    exec('def read(model, name):\n    return getattr(model, name)\n', namespace)
    return namespace['read']


@pytest.mark.parametrize('name', CLASSES)
def test_actual_prep_slot_fields_and_method_binding(name):
    cls = getattr(prep, name)
    model = cls(estimator=object()) if name == 'RFE' else cls()
    fields = object.__getattribute__(model, '__dict__')
    fields['user_note'] = object()
    assert vars(cls)['__getattribute__'] is owner._CONTEXT['descriptor']
    assert clean(read, model, '__dict__') is fields
    assert clean(read, model, '__class__') is cls
    assert clean(read, model, 'user_note') is fields['user_note']
    bound = clean(read, model, 'get_params')
    assert bound.__self__ is model
    # This audits the ordinary public expression, including its attribute read.
    def parameters(instance):
        return instance.get_params()
    expected = {key: object.__getattribute__(model, key) for key in cls._parameters}
    result = clean(parameters, model)
    assert list(result) == list(cls._parameters)
    assert all(result[key] is expected[key] for key in cls._parameters)
    if name not in CLASSIFIERS:
        bound = clean(read, model, 'fit_transform')
        assert bound.__self__ is model


@pytest.mark.parametrize('name', CLASSIFIERS)
def test_classifier_availability_and_instance_shadowing(name):
    cls = getattr(prep, name)
    model = object.__new__(cls)
    fields = object.__getattribute__(model, '__dict__')
    error = clean(read, model, 'fit_transform', error=AttributeError)
    assert f"This '{name}' has no attribute 'fit_transform'" in str(error)
    fields['transform'] = None  # hasattr tests presence, never callability/truth.
    bound = clean(read, model, 'fit_transform')
    assert bound.__self__ is model
    fields['fit_transform'] = object()
    assert clean(read, model, 'fit_transform') is fields['fit_transform']
    del fields['fit_transform'], fields['transform']
    clean(read, model, 'fit_transform', error=AttributeError)


def test_availability_subclass_method_binding_never_executes_method():
    calls = []
    class Child(prep.GaussianNB):
        def transform(self, value):
            calls.append('transform')
    model = object.__new__(Child)
    bound = clean(read, model, 'fit_transform')
    assert bound.__self__ is model and not calls


def test_availability_refuses_python_property_before_check_or_hook():
    calls = []
    class Child(prep.GaussianNB):
        @property
        def transform(self):
            calls.append('transform property')
            return None
    model = object.__new__(Child)
    clean(read, model, 'fit_transform', error=TypeError)
    assert not calls


def test_changed_availability_predicate_refuses_without_callback(monkeypatch):
    calls = []
    plan = owner._CONTEXT['plans'][prep.GaussianNB]['availability']['fit_transform']
    monkeypatch.setattr(plan['member'], 'check', lambda model: calls.append(model))
    clean(read, object.__new__(prep.GaussianNB), 'fit_transform', error=TypeError)
    assert not calls


def test_changed_predicate_code_refuses_without_running_it(monkeypatch):
    plan = owner._CONTEXT['plans'][prep.GaussianNB]['availability']['fit_transform']
    def changed(model):
        raise AssertionError('changed Python predicate ran')
    monkeypatch.setattr(plan['check'], '__code__', changed.__code__)
    clean(read, object.__new__(prep.GaussianNB), 'fit_transform', error=TypeError)


def test_changed_descriptor_hook_refuses_without_running_it(monkeypatch):
    calls = []
    plan = owner._CONTEXT['plans'][prep.GaussianNB]['availability']['fit_transform']
    monkeypatch.setattr(type(plan['member']), '__get__', lambda *args: calls.append(args))
    clean(read, object.__new__(prep.GaussianNB), 'fit_transform', error=TypeError)
    assert not calls


def test_python_function_get_metadata_is_never_used_as_a_descriptor():
    calls = []
    class Child(prep.GaussianNB):
        def transform(self, value):
            calls.append('method')
    function = vars(Child)['transform']
    function.__get__ = lambda *args: calls.append('fake function getter')
    model = object.__new__(Child)
    bound = clean(read, model, 'transform')
    assert bound.__self__ is model and bound.__func__ is function
    assert clean(read, model, 'fit_transform').__self__ is model
    assert not calls


def test_public_output_view_preserves_lifetime_and_readonly_storage():
    np = pytest.importorskip('numpy')
    storage = np.array([[1, 2], [3, 4]], dtype='f4')
    storage.flags.writeable = False
    reference = weakref.ref(storage)
    value = Array.from_buffer(storage)
    model = object.__new__(prep.GaussianNB)
    fields = object.__getattribute__(model, '__dict__')
    fields.update(theta_=value, _theta=value)
    output = clean(public_reader(), model, 'theta_')
    assert type(output) is np.ndarray and np.shares_memory(output, storage)
    assert not output.flags.writeable
    assert clean(public_reader(), model, '_theta') is value
    assert clean(public_reader(__mojolearn_internal__=True), model, 'theta_') is value
    del model, value, fields, storage
    gc.collect()
    assert reference() is not None
    np.testing.assert_array_equal(output, [[1, 2], [3, 4]])
    del output
    gc.collect()
    assert reference() is None
