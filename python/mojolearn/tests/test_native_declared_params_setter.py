"""Declared setters preserve shallow ordered mutation without Python hooks.

Constructors and bound-method lookup are test setup. The audit covers the
public setter operation, including ownership released by each assignment.
"""
import array
import weakref

import pytest

import mojolearn as ml
from mojolearn import _expansion_prep as prep, _expansion_decomp as decomp
from mojolearn._verify_native_runtime import audit_native_call

PREP_CLASSES = (
    'BernoulliNB', 'Binarizer', 'CategoricalNB', 'ComplementNB', 'GaussianNB',
    'IterativeImputer', 'KBinsDiscretizer', 'LabelBinarizer', 'LabelEncoder',
    'MultiLabelBinarizer', 'MultinomialNB', 'Normalizer', 'OneHotEncoder',
    'OrdinalEncoder', 'PolynomialFeatures', 'PowerTransformer',
    'QuadraticDiscriminantAnalysis', 'QuantileTransformer', 'RFE', 'SelectKBest',
    'SimpleImputer', 'SplineTransformer', 'TargetEncoder', 'VarianceThreshold',
)
DECOMP_CLASSES = (
    'AlternatingLeastSquares', 'CCA', 'ClassicalMDS', 'DictionaryLearning',
    'EllipticEnvelope', 'FactorAnalysis', 'FastICA', 'GaussianRandomProjection',
    'IncrementalPCA', 'Isomap', 'LatentDirichletAllocation',
    'LocallyLinearEmbedding', 'MDS', 'MinCovDet', 'MiniBatchDictionaryLearning',
    'MiniBatchSparsePCA', 'NMF', 'PLSCanonical', 'PLSRegression', 'SparseCoder',
    'SparsePCA', 'SparseRandomProjection',
)
CASES = [('prep', name) for name in PREP_CLASSES] + [('decomp', name) for name in DECOMP_CLASSES]


def clean(operation, *args, expected_error=None, **kwargs):
    audit = audit_native_call(operation, *args, **kwargs)
    report = audit.report()
    assert not any(report[key] for key in ('python_frames', 'forbidden_native_calls',
        'unresolved_native_calls', 'wrapper_work', 'unobserved_existing_threads')), report
    if expected_error is not None:
        assert isinstance(audit.exception, expected_error), report
        return audit.exception
    return audit.assert_native()


def make_model(family, name):
    cls = getattr(prep if family == 'prep' else decomp, name)
    if name == 'RFE':
        return cls(ml.Ridge())
    if name == 'SparseCoder':
        return cls(array.array('f', [1.0, 0.0]))
    return cls()


def example(family):
    return (prep.Normalizer(), 'norm', 'copy') if family == 'prep' else (decomp.ClassicalMDS(), 'metric', 'n_components')


@pytest.mark.parametrize('family,name', CASES)
def test_declared_setter_preserves_identity_fitted_state_and_unrelated_fields(family, name):
    model = make_model(family, name)
    # Use a scalar parameter for the broad family matrix; estimator and native
    # score-function replacements have dedicated ownership probes below.
    key = next(key for key in type(model)._parameters
               if type(object.__getattribute__(model, key)) in (type(None), str, int, float, bool))
    old = object.__getattribute__(model, key)
    fitted = array.array('f', [3.0, 4.0])
    user = object()
    model.__dict__.update(example_fitted_=fitted, user_metadata=user, _mojolearn_fitted=True)
    before = dict(model.__dict__)
    operation = model.set_params
    assert clean(operation) is model
    assert all(model.__dict__[key] is value for key, value in before.items())
    value = array.array('f', [1.0, 2.0])
    assert clean(operation, **{key: value}) is model
    assert object.__getattribute__(model, key) is value
    assert model.__dict__['example_fitted_'] is fitted
    assert model.__dict__['user_metadata'] is user
    assert model.__dict__['_mojolearn_fitted'] is True
    assert '_mojolearn_params_reset' not in model.__dict__
    assert clean(operation, **{key: old}) is model
    assert model.__dict__.keys() == before.keys()
    assert all(model.__dict__[key] is value for key, value in before.items())


@pytest.mark.parametrize('family', ['prep', 'decomp'])
def test_later_invalid_keyword_preserves_earlier_mutation_and_message(family):
    model, first, last = example(family)
    previous_last = object.__getattribute__(model, last)
    value = object()
    error = clean(model.set_params, **{first: value, 'unknown': 1, last: object()}, expected_error=ValueError)
    expected = (f"mojolearn: invalid parameter 'unknown' for {type(model).__name__}" if family == 'prep'
                else f"Invalid {type(model).__name__} parameter: unknown")
    assert str(error) == expected
    assert object.__getattribute__(model, first) is value
    assert object.__getattribute__(model, last) is previous_last


@pytest.mark.parametrize('family', ['prep', 'decomp'])
def test_nested_names_are_invalid_without_calling_nested_setters(family):
    calls = []

    class Callback:
        def set_params(self, **kwargs):
            calls.append(kwargs)

    model = prep.IterativeImputer(estimator=Callback()) if family == 'prep' else decomp.ClassicalMDS(metric_params=Callback())
    name = 'estimator' if family == 'prep' else 'metric_params'
    error = clean(model.set_params, **{name + '__value': 1}, expected_error=ValueError)
    assert name + '__value' in str(error) and not calls


@pytest.mark.parametrize('family', ['prep', 'decomp'])
def test_missing_parameter_can_be_restored_without_reading_other_fields(family):
    model, name, unused = example(family)
    del model.__dict__[name]
    del model.__dict__[unused]
    value = object()
    assert clean(model.set_params, **{name: value}) is model
    assert model.__dict__[name] is value and unused not in model.__dict__


@pytest.mark.parametrize('family', ['prep', 'decomp'])
def test_passive_subclass_keeps_the_builtin_assignment_contract(family):
    parent = prep.Normalizer if family == 'prep' else decomp.ClassicalMDS

    class Passive(parent):
        pass

    model = Passive()
    name = 'norm' if family == 'prep' else 'metric'
    value = object()
    assert clean(model.set_params, **{name: value}) is model
    assert model.__dict__[name] is value


@pytest.mark.parametrize('family', ['prep', 'decomp'])
def test_changed_parameter_descriptor_is_refused_before_hook(family, monkeypatch):
    model, name, _ = example(family)
    operation = model.set_params
    calls = []
    previous = model.__dict__[name]
    with monkeypatch.context() as patch:
        patch.setattr(type(model), name, property(lambda self: None,
                      lambda self, value: calls.append(value)), raising=False)
        clean(operation, **{name: 3}, expected_error=TypeError)
    assert not calls and model.__dict__[name] is previous


@pytest.mark.parametrize('family', ['prep', 'decomp'])
def test_custom_setattr_is_refused_before_hook(family):
    parent = prep.Normalizer if family == 'prep' else decomp.ClassicalMDS
    name = 'norm' if family == 'prep' else 'metric'
    calls = []

    class Callback(parent):
        def __setattr__(self, name, value):
            calls.append((name, value))

    model = object.__new__(Callback)
    base = prep._PrepBase if family == 'prep' else decomp._Base
    clean(base.set_params, model, **{name: 3}, expected_error=TypeError)
    assert not calls and object.__getattribute__(model, '__dict__') == {}


@pytest.mark.parametrize('family', ['prep', 'decomp'])
def test_dictionary_subclass_is_refused_without_iteration(family):
    model, name, _ = example(family)
    operation = model.set_params
    calls = []

    class Fields(dict):
        def __iter__(self):
            calls.append('iter')
            return super().__iter__()

        def keys(self):
            calls.append('keys')
            return super().keys()

    original = model.__dict__
    model.__dict__ = Fields(original)
    try:
        clean(operation, **{name: 3}, expected_error=TypeError)
        assert not calls
    finally:
        model.__dict__ = original


def test_replacing_python_finalizer_owner_refuses_but_same_reference_is_noop():
    calls = []

    class Finalizer:
        def __del__(self):
            calls.append('del')

    value = Finalizer()
    model = prep.IterativeImputer(estimator=value)
    assert clean(model.set_params, estimator=value) is model
    clean(model.set_params, estimator=None, expected_error=TypeError)
    assert model.__dict__['estimator'] is value and not calls


def test_weakref_callback_owner_is_refused_before_replacement():
    calls = []

    class Value:
        pass

    value = Value()
    retained_ref = weakref.ref(value, lambda ref: calls.append('weakref'))
    model = decomp.ClassicalMDS(metric_params=value)
    clean(model.set_params, metric_params=None, expected_error=TypeError)
    assert model.__dict__['metric_params'] is value and retained_ref() is value and not calls


@pytest.mark.parametrize('name', ['RFE', 'IterativeImputer'])
def test_builtin_estimator_parameter_can_be_replaced(name):
    new = ml.BayesianRidge()
    # No application reference keeps the previous estimator alive. Its last
    # owner can be released during the audited native assignment.
    model = prep.RFE(ml.Ridge()) if name == 'RFE' else prep.IterativeImputer(estimator=ml.Ridge())
    assert clean(model.set_params, estimator=new) is model
    assert model.__dict__['estimator'] is new


def test_fitted_builtin_estimator_parameter_can_be_replaced():
    import numpy as np
    x = np.arange(24, dtype=np.float32).reshape(8, 3) / 10
    model = prep.RFE(ml.Ridge().fit(x, np.arange(8, dtype=np.float32)))
    replacement = ml.BayesianRidge()
    assert clean(model.set_params, estimator=replacement) is model
    assert model.__dict__['estimator'] is replacement


def test_connected_score_functions_can_be_replaced_without_calls():
    model = prep.SelectKBest()
    for function in (prep.f_regression, prep.chi2, prep.f_classif):
        assert clean(model.set_params, score_func=function) is model
        assert model.__dict__['score_func'] is function


@pytest.mark.parametrize('name', ['RFE', 'IterativeImputer', 'GaussianRandomProjection'])
def test_successful_fit_state_and_transform_survive_a_parameter_update(name):
    import numpy as np
    x = np.arange(24, dtype=np.float32).reshape(8, 3) / 10
    if name == 'RFE':
        model = prep.RFE(ml.Ridge(), n_features_to_select=2).fit(x, np.arange(8, dtype=np.float32))
        update = {'step': 0.5}
    elif name == 'IterativeImputer':
        x[2, 1] = np.nan
        model = prep.IterativeImputer(max_iter=1, random_state=0).fit(x)
        update = {'verbose': 1}
    else:
        model = decomp.GaussianRandomProjection(n_components=2, random_state=0).fit(x)
        update = {'eps': 0.2}
    expected = model.transform(x)
    before = dict(model.__dict__)
    assert clean(model.set_params, **update) is model
    assert all(model.__dict__[key] is value for key, value in before.items() if key not in update)
    np.testing.assert_array_equal(model.transform(x), expected)
