# SPDX-License-Identifier: Apache-2.0
"""The prep owners never call back into Python during fit, transform or predict.

SimpleImputer, IterativeImputer, RFE, TargetEncoder and SelectKBest are one
native call each. A wrapped estimator (IterativeImputer, RFE) is cloned,
fitted and asked to predict through its connected native plan
(bindings/prep_native_estimator.mojo), a mojolearn cv splitter
(TargetEncoder) yields its folds from the model-selection owner directly, and
a Python callable (SimpleImputer strategy, SelectKBest score_func) is refused.

The probes record every Python frame entered during the operation
(`sys.setprofile`) and assert that none belongs to the wrapped estimator's or
splitter's class (its fit, predict, get_params, __init__, ...) or to a user
callable. Monkeypatching the class would itself be refused by the plan's
method-identity check, so the probe watches frames instead.

The source checks run without bindings; the probes need the built bindings."""
import ast
import inspect
import pathlib
import sys
import types

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
BINDINGS = ROOT / "bindings"

try:
    import mojolearn as ml
    from mojolearn.model_selection import KFold
    ml.Ridge  # noqa: B018  (the bindings loaded)
except Exception:  # no built bindings: only the source checks run
    ml = None
needs_bindings = pytest.mark.skipif(ml is None, reason="needs the built mojolearn bindings")


# ---------------------------------------------------------------- source checks (no bindings)
def _source(name):
    return (BINDINGS / name).read_text()


def test_iterimp_door_has_no_python_method_calls():
    src = _source("prep_iterimp_native.mojo")
    for needle in ('_method(', '"get_params"', 'import_module("inspect")', 'deepcopy', '"predict", Python.tuple(xm), keywords',
                   'def _user_call', 'def _clone'):
        assert needle not in src, needle
    assert src.count("prep_estimator_clone(estimator, context)") == 1
    assert 'prep_estimator_call(est, "fit"' in src and 'prep_estimator_call(est, "predict"' in src
    assert 'prep_estimator_admit(estimator, "IterativeImputer", context)' in src


def test_rfe_door_has_no_python_method_calls():
    src = _source("prep_select_native.mojo")
    for needle in ('_attr(_clone(', '_attr(est, "get_params")', '_attr(_attr(model, "estimator_"), operation)',
                   'def _clone', 'def _admit_estimator'):
        assert needle not in src, needle
    assert src.count("_fit_clone(estimator,") == 2
    assert 'prep_estimator_admit(estimator, "RFE", context)' in src
    assert "prep_estimator_call(est, operation" in src


def test_encode_door_calls_no_user_callable():
    src = _source("prep_encode_native.mojo")
    for needle in ("py_call_typed(strategy", "py_call(b.callable, strategy)", "py_call_typed(score_func",
                   "py_call(cv.split"):
        assert needle not in src, needle
    assert "prep_native_split(cv, x, y, context)" in src
    assert 'context["prep_encode_entry"]' in src


def test_model_selection_owner_serves_the_plan_operations():
    src = _source("model_selection_native.mojo")
    assert 'operation == "family_msel_estimator_call"' in src
    assert 'operation == "family_msel_estimator_clone"' in src
    assert "msel_native_clone(model, context)" in src
    helper = _source("prep_native_estimator.mojo")
    for op in ("family_msel_estimator_clone", "family_msel_estimator_call", "family_msel_split"):
        assert '"' + op + '"' in helper
    # the entry must be a compiled builtin, never a Python callable
    assert "py_type(entry) is not py_type(b.len)" in helper


def test_package_connects_the_entry_at_import():
    tree = ast.parse((ROOT / "python" / "mojolearn" / "__init__.py").read_text())
    keys = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "update":
            keys |= {kw.arg for kw in node.keywords}
    assert {"native_estimator_entry", "native_estimator_context"} <= keys
    prep = (ROOT / "python" / "mojolearn" / "_expansion_prep.py").read_text()
    assert "prep_encode_entry=_ENCODE_CALL" in prep
    assert 'mutual_info_classif: "mutual_info_classif"' in prep


# ---------------------------------------------------------------- probes (bindings)
def _class_codes(*classes):
    """The code objects of every Python function defined on the classes'
    MROs (methods, properties, static and class methods), object excluded."""
    codes = set()
    for cls in classes:
        for base in inspect.getmro(cls):
            if base is object:
                continue
            for value in vars(base).values():
                for fn in (value, getattr(value, "__func__", None), getattr(value, "fget", None)):
                    if isinstance(fn, types.FunctionType):
                        codes.add(fn.__code__)
    return codes


class _Probe:
    """Records the code object of every Python frame entered while active."""

    def __init__(self):
        self.codes = []

    def _hook(self, frame, event, arg):
        if event == "call":
            self.codes.append(frame.f_code)

    def __enter__(self):
        sys.setprofile(self._hook)
        return self

    def __exit__(self, *exc):
        sys.setprofile(None)
        return False

    def hits(self, forbidden):
        return sorted({c.co_qualname if hasattr(c, "co_qualname") else c.co_name
                       for c in self.codes if c in forbidden})


def _gaps(seed, n=120, d=5):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    X[:, 1] += X[:, 0]
    X[rng.random((n, d)) < 0.15] = np.nan
    return X


def _classes(seed=0, n=150, d=6):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    X = (rng.standard_normal((n, d)) + y[:, None] * np.linspace(0, 1.2, d)).astype(np.float32)
    return X, y


@needs_bindings
def test_simple_imputer_never_calls_a_strategy_callable():
    calls = []

    def strategy(column):
        calls.append(1)
        return 0.0

    with pytest.raises(ValueError, match="is not valid"):
        ml.SimpleImputer(strategy=strategy).fit(_gaps(0))
    assert calls == []
    forbidden = _class_codes(ml.SimpleImputer) - {ml.SimpleImputer.fit.__code__, ml.SimpleImputer.transform.__code__,
                                                  ml.SimpleImputer.fit_transform.__code__}
    for name in ("mean", "median", "most_frequent", "constant"):
        imp = ml.SimpleImputer(strategy=name, add_indicator=True)
        with _Probe() as probe:
            imp.fit(_gaps(1))
            imp.transform(_gaps(2))
        assert probe.hits(forbidden) == [], name


@needs_bindings
@pytest.mark.parametrize("kw", [dict(max_iter=3), dict(max_iter=2, n_nearest_features=3, random_state=2)])
def test_iterative_imputer_never_calls_the_estimator_python_methods(kw):
    X, Xh = _gaps(3), _gaps(4)
    forbidden = _class_codes(ml.Ridge)
    imp = ml.IterativeImputer(estimator=ml.Ridge(alpha=2.0), **kw)
    with _Probe() as probe:
        out = imp.fit_transform(X)
        again = imp.transform(Xh)
    assert probe.hits(forbidden) == []
    assert np.isfinite(np.asarray(out)).all() and np.isfinite(np.asarray(again)).all()
    for _, _, est in imp.imputation_sequence_:
        assert type(est) is ml.Ridge


@needs_bindings
def test_iterative_imputer_refuses_without_a_native_plan():
    class MyRidge(ml.Ridge):
        pass

    with pytest.raises(TypeError, match="IterativeImputer requires a mojolearn estimator; got MyRidge"):
        ml.IterativeImputer(estimator=MyRidge()).fit(_gaps(5))
    with pytest.raises(ValueError, match="return_std"):
        ml.IterativeImputer(estimator=ml.Ridge(), sample_posterior=True).fit(_gaps(5))


@needs_bindings
@pytest.mark.parametrize("make", ["lda", "ridge"])
def test_rfe_never_calls_the_estimator_python_methods(make):
    X, y = _classes()
    if make == "lda":
        est, cls, target = ml.LinearDiscriminantAnalysis(), ml.LinearDiscriminantAnalysis, y
    else:
        est, cls, target = ml.Ridge(), ml.Ridge, X[:, 0] * 2 + 0.1 * y.astype(np.float32)
    forbidden = _class_codes(cls)
    rfe = ml.RFE(est, n_features_to_select=3, step=1)
    with _Probe() as probe:
        rfe.fit(X, target)
        rfe.predict(X)
        rfe.score(X, target)
        if make == "lda":
            rfe.predict_proba(X)
            rfe.decision_function(X)
    assert probe.hits(forbidden) == []
    assert type(rfe.estimator_) is cls and rfe.n_features_ == 3


@needs_bindings
def test_rfe_refusals_without_a_native_plan():
    X, y = _classes()

    class MyLDA(ml.LinearDiscriminantAnalysis):
        pass

    # The native LDA constructor now rejects the unsupported subclass before
    # RFE can admit an estimator. Preserve that earlier no-callback refusal.
    with pytest.raises(TypeError, match="native LDA requires its connected class without Python subclass hooks"):
        ml.RFE(MyLDA()).fit(X, y)
    with pytest.raises(TypeError, match="unexpected keyword argument 'sample_weight'"):
        ml.RFE(ml.LinearDiscriminantAnalysis()).fit(X, y, sample_weight=np.ones(len(y), np.float32))
    fitted = ml.RFE(ml.Ridge(), n_features_to_select=2).fit(X, X[:, 0])
    with pytest.raises(AttributeError, match="predict_proba"):
        fitted.predict_proba(X)


@needs_bindings
def test_target_encoder_never_calls_the_splitter_split_method():
    rng = np.random.default_rng(7)
    X = rng.integers(0, 4, (200, 2)).astype(np.int64)
    y = (X[:, 0] + rng.random(200) > 2).astype(np.int64)
    cv = KFold(4)
    forbidden = _class_codes(type(cv))
    enc = ml.TargetEncoder(cv=cv, smooth=2.0)
    with _Probe() as probe:
        out = enc.fit_transform(X, y)
    assert probe.hits(forbidden) == []
    pairs = list(KFold(4).split(X, y))
    np.testing.assert_array_equal(np.asarray(out), np.asarray(ml.TargetEncoder(cv=pairs, smooth=2.0).fit_transform(X, y)))


@needs_bindings
def test_select_kbest_refuses_user_score_functions_and_runs_mutual_info_natively():
    X, y = _classes()
    calls = []

    def score(X, y):
        calls.append(1)
        return np.ones(X.shape[1])

    with pytest.raises(TypeError, match="a user callable is not supported"):
        ml.SelectKBest(score, k=2).fit(X, y)
    assert calls == []
    forbidden = {ml.mutual_info_classif.__code__}
    sel = ml.SelectKBest(ml.mutual_info_classif, k=3)
    with _Probe() as probe:
        sel.fit(X, y)
    assert probe.hits(forbidden) == []
    np.testing.assert_array_equal(np.asarray(sel.scores_), np.asarray(ml.mutual_info_classif(X, y)))


@needs_bindings
@pytest.mark.parametrize('name,kwargs', [
    ('Ridge', {'alpha': 0.5}), ('BayesianRidge', {'max_iter': 20}),
    ('SVR', {'kernel': 'linear', 'max_iter': 100}), ('LinearSVR', {'max_iter': 100}),
    ('Lasso', {'alpha': 0.01}), ('ElasticNet', {'alpha': 0.01}),
])
def test_rfe_preserves_weighted_native_fits(name, kwargs):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(48, 4)).astype(np.float32)
    y = (X[:, 0] - 0.7 * X[:, 2] + 0.1 * rng.normal(size=48)).astype(np.float32)
    weights = np.linspace(0.25, 2, len(y), dtype=np.float32)
    cls = getattr(ml, name)
    estimator = cls(**kwargs)
    selector = ml.RFE(estimator, n_features_to_select=2)
    with _Probe() as probe:
        selector.fit(X, y, sample_weight=weights)
        actual = selector.predict(X)
    assert probe.hits(_class_codes(cls)) == []
    direct = cls(**kwargs).fit(X[:, np.asarray(selector.support_)], y, sample_weight=weights)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(direct.predict(X[:, np.asarray(selector.support_)])))


@needs_bindings
def test_rfe_preserves_native_forest_fit_keywords():
    X, y = _classes(n=48, d=4)
    selector = ml.RFE(ml.RandomForestClassifier(n_estimators=3, max_depth=3), n_features_to_select=2)
    with _Probe() as probe:
        selector.fit(X, y, sample_weight=np.ones(len(y), np.float32), feature_importances=True)
    assert probe.hits(_class_codes(ml.RandomForestClassifier)) == []
    assert np.asarray(selector.estimator_.feature_importances_).shape == (2,)


@needs_bindings
def test_iterative_imputer_preserves_supplied_native_posterior():
    from mojolearn.gaussian_process import ConstantKernel, RBF
    X = _gaps(41, n=24, d=3)
    kernel = ConstantKernel(1.0, 'fixed') * RBF(1.0, 'fixed')
    kwargs = dict(sample_posterior=True, max_iter=2, random_state=7, min_value=-10, max_value=10)
    estimator = ml.GaussianProcessRegressor(kernel=kernel, optimizer=None)
    imputer = ml.IterativeImputer(estimator=estimator, **kwargs)
    with _Probe() as probe:
        actual = imputer.fit_transform(X)
        transformed = imputer.transform(X)
    assert probe.hits(_class_codes(type(estimator), type(kernel), RBF, ConstantKernel)) == []
    twin = ml.IterativeImputer(estimator=ml.GaussianProcessRegressor(kernel=kernel, optimizer=None), **kwargs)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(twin.fit_transform(X)))
    assert np.isfinite(np.asarray(transformed)).all()
    for _, _, fitted in imputer.imputation_sequence_:
        assert fitted.kernel is not kernel
        assert fitted.kernel.a is not kernel.a
        assert fitted.kernel.b is not kernel.b


@needs_bindings
def test_native_composition_refuses_callback_keywords_without_executing_them():
    calls = []
    class Weight:
        def __float__(self):
            calls.append('float')
            return 1.0
        def __array__(self, *args, **kwargs):
            calls.append('array')
            return np.ones(48)
    X, y = _classes(n=48, d=4)
    with pytest.raises(TypeError):
        ml.RFE(ml.Ridge()).fit(X, y, sample_weight=Weight())
    assert calls == []


@needs_bindings
@pytest.mark.parametrize('name', [
    'VotingRegressor', 'VotingClassifier', 'StackingRegressor', 'StackingClassifier',
    'MultiOutputRegressor', 'MultiOutputClassifier', 'OneVsRestClassifier', 'CalibratedClassifierCV',
])
def test_meta_estimators_use_native_members_and_splitters(name):
    X, labels = _classes(seed=17, n=48, d=4)
    y = (X[:, 0] - 0.25 * X[:, 1]).astype(np.float32)
    classes = [ml.Ridge, ml.BayesianRidge, ml.LogisticRegression, ml.GaussianNB, ml.LinearSVC, KFold]
    cv = KFold(2, shuffle=True, random_state=1)
    if name in ('VotingRegressor', 'StackingRegressor'):
        members = [('ridge', ml.Ridge()), ('bayes', ml.BayesianRidge(max_iter=20))]
        estimator = ml.VotingRegressor(members) if name == 'VotingRegressor' else ml.StackingRegressor(members, cv=cv)
    elif name in ('VotingClassifier', 'StackingClassifier'):
        members = [('logistic', ml.LogisticRegression(max_iter=30)), ('bayes', ml.GaussianNB())]
        estimator = ml.VotingClassifier(members, voting='soft') if name == 'VotingClassifier' else ml.StackingClassifier(members, cv=cv)
        y = labels
    elif name == 'MultiOutputRegressor':
        estimator, y = ml.MultiOutputRegressor(ml.Ridge()), np.column_stack([y, -y])
    elif name == 'MultiOutputClassifier':
        estimator, y = ml.MultiOutputClassifier(ml.LogisticRegression(max_iter=30)), np.column_stack([labels, labels % 2])
    elif name == 'OneVsRestClassifier':
        estimator, y = ml.OneVsRestClassifier(ml.LogisticRegression(max_iter=30)), labels
    else:
        estimator, y = ml.CalibratedClassifierCV(cv=cv), labels
    forbidden = _class_codes(*classes)
    with _Probe() as probe:
        estimator.fit(X, y)
        output = estimator.predict(X)
        score = estimator.score(X, y)
    assert probe.hits(forbidden) == []
    assert np.asarray(output).shape == np.asarray(y).shape
    assert np.isfinite(float(score))


@needs_bindings
def test_meta_estimator_refuses_spoofed_members_without_callbacks():
    calls = []
    class Fake:
        __module__ = 'mojolearn'
        def get_params(self, **kwargs):
            calls.append('params')
            return {}
        def fit(self, *args, **kwargs):
            calls.append('fit')
            return self
        def predict(self, *args):
            calls.append('predict')
            return []
        def __repr__(self):
            calls.append('repr')
            return 'Fake'
    X, y = _classes(n=24, d=3)
    with pytest.raises(TypeError):
        ml.VotingRegressor([('fake', Fake())]).fit(X, y)
    assert calls == []
