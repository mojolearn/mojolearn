"""Native scorer ABI, option semantics and whole public CV audit coverage."""
import numpy as np
import pytest

import mojolearn as ml
from mojolearn import metrics
from mojolearn import model_selection as ms
from mojolearn._verify_native_runtime import audit_native_call


def data():
    x = np.arange(48, dtype='f4').reshape(16, 3) / np.float32(48)
    x[:, 1] *= x[:, 1]
    x[:, 2] = np.cos(x[:, 2])
    return x, (x @ np.array([.5, -.25, .75], 'f4')).astype('f4')


@pytest.mark.parametrize('name', ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error',
                                  'explained_variance', 'max_error'])
def test_named_scorer_public_cv_is_native(name):
    x, y = data()
    result = audit_native_call(ms.cross_val_score, ml.Ridge(), x, y, cv=3,
                               scoring=name, n_jobs=1, error_score='raise').assert_native()
    assert np.asarray(result).shape == (3,)
    assert np.isfinite(np.asarray(result)).all()


@pytest.mark.parametrize('cls', [ml.LinearRegression, ml.Ridge])
def test_public_make_scorer_and_call_use_native_metric_options(cls):
    x, y = data()
    model = cls().fit(x, y)
    scorer = audit_native_call(ms.make_scorer, metrics.mean_absolute_error,
                               greater_is_better=False, multioutput='uniform_average').assert_native()
    got = audit_native_call(scorer.__call__, model, x, y).assert_native()
    # The audit represents an external user even when pytest imports this
    # test as mojolearn.tests, whose direct calls are package-internal.
    public_prediction = np.asarray(audit_native_call(model.predict, x).assert_native())
    assert public_prediction.dtype == np.float64
    # Test oracle: Ridge's public output widens its float32 kernel result;
    # package-internal scoring consumes the original words, as before.
    kernel_prediction = public_prediction.astype('f4')
    np.testing.assert_array_equal(public_prediction, kernel_prediction.astype('f8'))
    with pytest.raises(TypeError, match='float32'):
        metrics.mean_absolute_error(y, public_prediction)
    expected = -float(metrics.mean_absolute_error(y, kernel_prediction))
    assert np.float64(got).view('u8') == np.float64(expected).view('u8')


def test_every_named_metric_has_a_registered_native_plan():
    table = ms._NATIVE_CONTEXT['family_msel_metric_plans']
    for scorer in ms._NATIVE_CONTEXT['family_msel_scorers'].values():
        assert vars(scorer)['_score_func'] in table


def test_every_fixed_metric_plan_connects_its_defining_module():
    from types import BuiltinFunctionType, FunctionType
    from mojolearn import _expansion_metrics, _metrics_impl
    from mojolearn._native_msel_metric_plans import METRIC_PLANS

    table = ms._NATIVE_CONTEXT['family_msel_metric_plans']
    for name, recipe in METRIC_PLANS.items():
        function = vars(_metrics_impl).get(name)
        if function is None:
            function = vars(_expansion_metrics)[name]
        assert type(function) is FunctionType
        plan = table[function]
        assert plan['operation'] == recipe['operation']
        assert plan['code'] is function.__code__
        assert plan['context'] is function.__globals__['_NATIVE_CONTEXT']
        assert plan['entry'] is function.__globals__['_NATIVE']
        assert type(plan['entry']) is BuiltinFunctionType


def test_expansion_only_metric_scorer_consumes_native_prediction():
    from mojolearn import _expansion_metrics

    x, y = data()
    model = ml.Ridge().fit(x, y)
    metric = _expansion_metrics.r2_score_options
    options = dict(sample_weight=None, multioutput='uniform_average', force_finite=True,
                   numeric_mode='identical')
    scorer = audit_native_call(ms.make_scorer, metric, **options).assert_native()
    got = audit_native_call(scorer.__call__, model, x, y).assert_native()
    public_prediction = np.asarray(audit_native_call(model.predict, x).assert_native())
    assert public_prediction.dtype == np.float64
    kernel_prediction = public_prediction.astype('f4')
    np.testing.assert_array_equal(public_prediction, kernel_prediction.astype('f8'))
    expected = float(metric(y, kernel_prediction, **options))
    assert np.float64(got).view('u8') == np.float64(expected).view('u8')


def test_custom_scorer_and_metric_callbacks_refuse_before_call():
    calls = []

    class Callback:
        def __call__(self, *args, **kwargs):
            calls.append('call')
        def __getattribute__(self, name):
            calls.append(name)
            return super().__getattribute__(name)

    for operation in (ms.get_scorer, ms.make_scorer):
        audit = audit_native_call(operation, Callback())
        assert isinstance(audit.exception, TypeError), audit.report()
        assert calls == []
        assert not audit.report()['python_frames'], audit.report()


@pytest.fixture(scope='module')
def named_scorer_models():
    # Testing setup and independent metric oracles stay outside the audited
    # public scorer call. Positive regression values cover log/deviance losses.
    x = np.arange(108, dtype='f4').reshape(36, 3) / np.float32(108)
    x[:, 1] = np.sin(x[:, 1] * np.float32(6))
    x[:, 2] *= x[:, 2]
    options = dict(n_estimators=3, max_depth=3, random_state=17, numeric_mode='identical')
    targets = {
        'regression': np.float32(2) + x[:, 0] * np.float32(.4) + x[:, 2] * np.float32(.3),
        'binary': (np.arange(36) % 2).astype('i4'),
        'multiclass': (np.arange(36) % 3).astype('i4'),
    }
    return {key: (ml.RandomForestRegressor(**options).fit(x, y)
                  if key == 'regression' else ml.RandomForestClassifier(**options).fit(x, y), x, y)
            for key, y in targets.items()}


def test_binary_top_k_scorer_consumes_positive_class_probability(named_scorer_models):
    model, x, y = named_scorer_models['binary']
    scorer = ms.get_scorer('top_k_accuracy')
    # Test setup chooses a nontrivial top-one result instead of the default
    # top-two score, which is always one for two classes.
    vars(scorer)['_kwargs']['k'] = 1
    got = audit_native_call(scorer.__call__, model, x, y).assert_native()
    probability = np.asarray(model.predict_proba(x))[:, 1].astype('f4')
    expected = metrics.top_k_accuracy_score(y, probability, k=1)
    assert np.float64(got).view('u8') == np.float64(expected).view('u8')


@pytest.mark.parametrize('name', sorted(ms._NATIVE_CONTEXT['family_msel_scorers']))
def test_every_named_scorer_consumes_native_prediction_and_matches_public_metric(name, named_scorer_models):
    scorer = ms.get_scorer(name)
    fields = vars(scorer)
    metric = fields['_score_func']
    regression = {
        'explained_variance_score', 'r2_score', 'max_error', 'median_absolute_error',
        'mean_absolute_error', 'mean_absolute_percentage_error', 'mean_squared_error',
        'mean_squared_log_error', 'root_mean_squared_error', 'root_mean_squared_log_error',
        'mean_poisson_deviance', 'mean_gamma_deviance', 'd2_absolute_error_score',
    }
    kind = ('regression' if metric.__name__ in regression else
            'multiclass' if name.startswith(('roc_auc_ovr', 'roc_auc_ovo')) else 'binary')
    model, x, y = named_scorer_models[kind]
    got = audit_native_call(scorer.__call__, model, x, y).assert_native()
    methods = fields['_response_method']
    methods = (methods,) if isinstance(methods, str) else methods
    method = next(method for method in methods if hasattr(model, method))
    pred = audit_native_call(getattr(model, method), x).assert_native()
    if kind == 'regression':
        # Public predictions widen the original float32 kernel words. As in
        # the make_scorer test above, the metric oracle consumes those words;
        # passing the public float64 view directly is an unsupported metric
        # dtype, not a scorer result. Prove narrowing loses no information.
        public_prediction = np.asarray(pred)
        pred = public_prediction.astype('f4')
        np.testing.assert_array_equal(public_prediction, pred.astype(public_prediction.dtype))
    if method == 'predict_proba' and fields['_name'] in ms._BINARY_PROBA:
        array = np.asarray(pred)
        if array.ndim == 2 and array.shape[1] == 2:
            pred = array[:, 1].astype('f4')
    expected = fields['_sign'] * float(metric(y, pred, **fields['_kwargs']))
    assert np.isfinite(got), (name, got)
    assert np.float64(got).view('u8') == np.float64(expected).view('u8'), (name, got, expected)
