# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""scikit-learn interoperability of every public mojolearn estimator.

Swapping `sklearn` for `mojolearn` in an import must keep working with
scikit-learn's own tools: `clone`, `Pipeline`, `GridSearchCV`,
`cross_val_score`, `is_classifier` / `is_regressor` and `check_estimator`.

Two parts:

1. STRUCTURAL (always runs, no binding needed, nothing is fitted): every
   public class with `fit` has `get_params` / `set_params` naming exactly its
   constructor parameters, clones, answers scikit-learn's tag protocol with
   the same estimator type as the scikit-learn class of the same name, sits
   in the module scikit-learn keeps it in, and builds inside a `Pipeline` and
   a `GridSearchCV`.
2. FITTING (opt-in, `MOJOLEARN_RUN_SKLEARN_CHECKS=1`, needs a device or CPU
   training binding): a `GridSearchCV` over a `Pipeline` and scikit-learn's
   `cross_val_score` actually run, and scikit-learn's `parametrize_with_checks`
   runs over every estimator that carries a scikit-learn class name, with the
   short expected-failure list below.
"""

import importlib
import inspect
import os

import pytest

import mojolearn
from mojolearn._mode import NumericModeMixin
from mojolearn._sklearn_modules import LAYOUT

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
from sklearn.base import clone, is_classifier, is_regressor  # noqa: E402
from sklearn.model_selection import GridSearchCV  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.utils import all_estimators, get_tags  # noqa: E402

RUN_FITS = os.environ.get("MOJOLEARN_RUN_SKLEARN_CHECKS") == "1"
FIT_REASON = ("fits on a device (or CPU training) binding: set MOJOLEARN_RUN_SKLEARN_CHECKS=1 "
              "on a machine that has one")

#: Public classes with `fit` that are not scikit-learn estimators and so have
#: no `get_params`.
NOT_SKLEARN_STYLE = {
    # torch-style layers and the CNN built from them (forward/backward
    # modules; not in the public docs, docs-site/api_exclude.txt)
    "AdaptiveAvgPool2d", "AdaptiveMaxPool2d", "AvgPool1d", "AvgPool2d", "BasicBlock", "BatchNorm1d",
    "BatchNorm2d", "CNNClassifier", "Conv1d", "Conv2d", "Dropout2d", "MaxPool1d", "MaxPool2d",
    # statsmodels-style: the series is a constructor argument, as in statsmodels
    "ExponentialSmoothing",
}

#: Public names left out of the docs as unverified (docs-site/api_exclude.txt).
UNVERIFIED = {"ExperimentalTwoLevelFeatureFreq", "OrderedRMSE"}

#: Estimators without a default `score`: scikit-learn's MultiOutputClassifier
#: scores subset accuracy over 2-D labels, which the metrics binding does not
#: compute (multilabel targets are refused by name), so pass `scoring=`.
NO_DEFAULT_SCORE = {"MultiOutputClassifier"}

#: The modules scikit-learn has that are new files here (re-exports).
NEW_MODULES = {
    "calibration": ("CalibratedClassifierCV",),
    "covariance": ("EllipticEnvelope", "MinCovDet"),
    "cross_decomposition": ("CCA", "PLSCanonical", "PLSRegression"),
    "discriminant_analysis": ("LinearDiscriminantAnalysis", "QuadraticDiscriminantAnalysis"),
    "feature_selection": ("RFE", "SelectKBest", "VarianceThreshold", "chi2", "f_classif", "f_regression",
                          "mutual_info_classif", "mutual_info_regression", "r_regression"),
    "gaussian_process": ("GaussianProcessClassifier", "GaussianProcessRegressor", "ConstantKernel", "Matern",
                         "RBF", "WhiteKernel"),
    "impute": ("IterativeImputer", "KNNImputer", "SimpleImputer"),
    "isotonic": ("IsotonicRegression",),
    "kernel_approximation": ("AdditiveChi2Sampler", "Nystroem", "PolynomialCountSketch", "RBFSampler",
                             "SkewedChi2Sampler"),
    "kernel_ridge": ("KernelRidge",),
    "multiclass": ("OneVsRestClassifier",),
    "multioutput": ("MultiOutputClassifier", "MultiOutputRegressor"),
    "naive_bayes": ("BernoulliNB", "CategoricalNB", "ComplementNB", "GaussianNB", "MultinomialNB"),
    "random_projection": ("GaussianRandomProjection", "SparseRandomProjection"),
    "semi_supervised": ("LabelPropagation", "LabelSpreading"),
    "tree": ("DecisionTreeClassifier", "DecisionTreeRegressor"),
}


def _factories():
    """Constructors for the classes whose `__init__` has required arguments."""
    ml = mojolearn
    rng = np.random.default_rng(0)
    dictionary = rng.random((5, 8), dtype=np.float32)
    series = rng.random(42, dtype=np.float32)
    series2d = rng.random((42, 2), dtype=np.float32)
    return {
        "MultiOutputClassifier": lambda: ml.MultiOutputClassifier(ml.DecisionTreeClassifier()),
        "MultiOutputRegressor": lambda: ml.MultiOutputRegressor(ml.Ridge()),
        "OneVsRestClassifier": lambda: ml.OneVsRestClassifier(ml.LogisticRegression()),
        "RFE": lambda: ml.RFE(ml.LinearDiscriminantAnalysis(), n_features_to_select=2),
        "SparseCoder": lambda: ml.SparseCoder(dictionary, transform_n_nonzero_coefs=2),
        "StackingClassifier": lambda: ml.StackingClassifier(
            [("rf", ml.RandomForestClassifier()), ("dt", ml.DecisionTreeClassifier())]),
        "StackingRegressor": lambda: ml.StackingRegressor(
            [("rf", ml.RandomForestRegressor()), ("dt", ml.DecisionTreeRegressor())]),
        "VotingClassifier": lambda: ml.VotingClassifier(
            [("rf", ml.RandomForestClassifier()), ("dt", ml.DecisionTreeClassifier())], voting="soft"),
        "VotingRegressor": lambda: ml.VotingRegressor(
            [("rf", ml.RandomForestRegressor()), ("dt", ml.DecisionTreeRegressor())]),
        # statsmodels-style: the series is the first constructor argument
        "AutoARIMA": lambda: ml.AutoARIMA(series),
        "STL": lambda: ml.STL(series, period=7),
        "VAR": lambda: ml.VAR(series2d),
        "IVFIndex": lambda: ml.IVFIndex(n_lists=4, n_probes=2),
        "IVFPQIndex": lambda: ml.IVFPQIndex(n_lists=4, n_probes=2),
        "IVFSQIndex": lambda: ml.IVFSQIndex(n_lists=4, n_probes=2),
        "IVFRaBitQIndex": lambda: ml.IVFRaBitQIndex(n_lists=4, n_probes=2),
    }


def _public_fit_classes():
    out = {}
    for name in mojolearn.__all__:
        try:
            obj = getattr(mojolearn, name)
        except (ImportError, OSError, AttributeError):
            continue  # an optional CPU-inference door absent from this install
        if inspect.isclass(obj) and callable(getattr(obj, "fit", None)):
            out[name] = obj
    return out


PUBLIC = _public_fit_classes()
ESTIMATOR_NAMES = sorted(n for n, c in PUBLIC.items()
                         if callable(getattr(c, "get_params", None)) and n not in UNVERIFIED)
SKLEARN = {n: c for n, c in all_estimators()}
SHARED_NAMES = sorted(n for n in ESTIMATOR_NAMES if n in SKLEARN)


def _make(name):
    factory = _factories().get(name)
    return factory() if factory is not None else PUBLIC[name]()


def _constructor_params(cls):
    params = inspect.signature(cls.__init__).parameters.values()
    names = {p.name for p in params
             if p.name != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)}
    if issubclass(cls, NumericModeMixin):
        names.add("numeric_mode")
    return names


# ----------------------------------------------------------------- structural

def test_every_class_with_fit_reports_its_parameters():
    missing = sorted(n for n, c in PUBLIC.items()
                     if not callable(getattr(c, "get_params", None)) and n not in NOT_SKLEARN_STYLE)
    assert not missing, f"public classes with fit but no get_params: {missing}"


@pytest.mark.parametrize("name", ESTIMATOR_NAMES)
def test_params_clone_and_tags(name):
    est = _make(name)
    params = est.get_params(deep=False)
    assert set(params) == _constructor_params(type(est)), name
    assert est.set_params(**params) is est
    twin = clone(est)
    assert type(twin) is type(est)
    assert set(twin.get_params(deep=False)) == set(params)
    tags = get_tags(est)
    assert tags.estimator_type in (None, "classifier", "regressor", "clusterer", "outlier_detector",
                                   "density_estimator")
    if callable(getattr(est, "transform", None)):
        assert tags.transformer_tags is not None, name
    if tags.estimator_type in ("classifier", "regressor"):
        assert tags.target_tags.required, name
        if name not in NO_DEFAULT_SCORE:
            assert callable(getattr(est, "score", None)), f"{name} has no default score for GridSearchCV"


@pytest.mark.parametrize("name", SHARED_NAMES)
def test_estimator_type_matches_scikit_learn(name):
    ours = _make(name)
    theirs_cls = SKLEARN[name]
    try:
        theirs = theirs_cls()
    except TypeError:
        theirs = None  # a scikit-learn meta-estimator with required arguments
    kind = get_tags(ours).estimator_type
    if theirs is not None:
        assert kind == get_tags(theirs).estimator_type, name
    assert is_classifier(ours) == (kind == "classifier")
    assert is_regressor(ours) == (kind == "regressor")


@pytest.mark.parametrize("name", SHARED_NAMES)
def test_estimator_is_in_scikit_learns_module(name):
    module = SKLEARN[name].__module__.split(".")[1]
    found = getattr(importlib.import_module("mojolearn." + module), name)
    assert found is getattr(mojolearn, name), (module, name)


@pytest.mark.parametrize("module", sorted(set(LAYOUT) | set(NEW_MODULES)))
def test_module_layout_names_resolve_to_the_package_objects(module):
    mod = importlib.import_module("mojolearn." + module)
    for name in LAYOUT.get(module, ()) + NEW_MODULES.get(module, ()):
        assert getattr(mod, name) is getattr(mojolearn, name), (module, name)
    assert "__all__" not in vars(mod) or set(LAYOUT.get(module, ()) + NEW_MODULES.get(module, ())) <= set(mod.__all__)
    with pytest.raises(AttributeError):
        getattr(mod, "NotAnEstimator")


@pytest.mark.parametrize("name", ESTIMATOR_NAMES)
def test_pipeline_and_grid_search_construct(name):
    est = _make(name)
    params = est.get_params(deep=False)
    pipe = Pipeline([("est", est)])
    deep = pipe.get_params(deep=True)
    assert all(f"est__{k}" in deep for k in params), name
    assert is_classifier(pipe) == is_classifier(est)
    assert is_regressor(pipe) == is_regressor(est)
    key = sorted(params)[0] if params else None
    grid = GridSearchCV(pipe, {f"est__{key}": [params[key]]} if key else {})
    assert type(clone(grid).estimator.named_steps["est"]) is type(est)


# ------------------------------------------------- before fit (no binding)

#: scikit-learn's estimator checks that never fit: parameters are stored
#: verbatim by the constructor and checked at fit, and every predict,
#: transform and score called before fit raises NotFittedError.
NO_FIT_CHECKS = ("check_no_attributes_set_in_init", "check_do_not_raise_errors_in_init_or_set_params",
                 "check_parameters_default_constructible", "check_get_params_invariance", "check_set_params",
                 "check_estimators_unfitted", "check_transformers_unfitted")

#: The methods that need a fitted estimator.
NEEDS_FIT = ("predict", "predict_proba", "predict_log_proba", "decision_function", "transform",
             "inverse_transform", "score", "score_samples")


@pytest.mark.parametrize("check", NO_FIT_CHECKS)
@pytest.mark.parametrize("name", SHARED_NAMES)
def test_scikit_learn_checks_that_need_no_fit(name, check):
    from sklearn.utils import estimator_checks
    est = _make(name)
    if check == "check_transformers_unfitted" and not (
            callable(getattr(est, "transform", None)) and get_tags(est).requires_fit):
        pytest.skip("not a transformer that needs a fit")
    getattr(estimator_checks, check)(name, est)


#: Not scikit-learn estimators, and their constructors set public attributes
#: ending in an underscore, which scikit-learn's fitted-state rule reads as
#: fitted. Each says which.
UNFITTED_EXEMPT = {
    # cuML's ARIMA: k_, n_exog_ and complexity_ resolve from order and trend in the constructor
    "ARIMA",
}


@pytest.mark.parametrize("name", sorted(set(ESTIMATOR_NAMES) - UNFITTED_EXEMPT))
def test_unfitted_calls_raise_not_fitted_error(name):
    from sklearn.exceptions import NotFittedError as SklearnNotFittedError
    from sklearn.utils.validation import check_is_fitted

    from mojolearn.exceptions import NotFittedError
    est = _make(name)
    if not get_tags(est).requires_fit:
        pytest.skip("stateless: usable without fit")
    with pytest.raises(SklearnNotFittedError):
        check_is_fitted(est)
    X = np.ones((4, 3), dtype=np.float32)
    y = np.zeros(4, dtype=np.float32)
    for method in NEEDS_FIT:
        if not callable(getattr(est, method, None)):
            continue
        args = (X, y) if method == "score" else (X,)
        with pytest.raises(NotFittedError) as caught:
            getattr(est, method)(*args)
        assert isinstance(caught.value, (SklearnNotFittedError, ValueError, AttributeError)), (name, method)


def test_not_fitted_error_needs_no_scikit_learn():
    import subprocess
    import sys
    root = os.path.dirname(os.path.dirname(mojolearn.__file__))
    code = ("import sys; sys.path.insert(0, %r); import mojolearn.exceptions as E; "
            "assert 'sklearn' not in sys.modules; "
            "assert issubclass(E.NotFittedError, ValueError) and issubclass(E.NotFittedError, AttributeError)" % root)
    subprocess.run([sys.executable, "-c", code], check=True)


# ------------------------------------------------------------------ fitting

def _small_problem():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 4)).astype(np.float32)
    y_class = (X[:, 0] + X[:, 1] > 0).astype(np.int64)
    y_reg = (X @ np.array([1.0, -2.0, 0.5, 0.0], dtype=np.float32)).astype(np.float32)
    return X, y_class, y_reg


@pytest.mark.skipif(not RUN_FITS, reason=FIT_REASON)
def test_grid_search_over_a_pipeline_fits():
    X, y, _ = _small_problem()
    pipe = Pipeline([("scale", mojolearn.StandardScaler()), ("clf", mojolearn.LogisticRegression())])
    search = GridSearchCV(pipe, {"clf__C": [0.1, 1.0]}, cv=3).fit(X, y)
    assert search.best_params_["clf__C"] in (0.1, 1.0)
    assert 0.0 <= search.best_score_ <= 1.0


@pytest.mark.skipif(not RUN_FITS, reason=FIT_REASON)
def test_scikit_learn_cross_val_score_runs_on_mojolearn_estimators():
    from sklearn.model_selection import cross_val_score
    X, y_class, y_reg = _small_problem()
    reg = cross_val_score(mojolearn.Ridge(), X, y_reg, cv=3)
    clf = cross_val_score(mojolearn.GaussianNB(), X, y_class, cv=3)
    assert len(reg) == 3 and len(clf) == 3
    assert all(0.0 <= s <= 1.0 for s in clf)


# scikit-learn's own estimator checks. They run on every estimator that
# carries a scikit-learn class name, except FLOAT32_ONLY.
#
#: Estimators whose documented input contract is float32 only: a float64
#: array raises TypeError. scikit-learn's checks generate float64 data, so
#: nearly every fitting check would fail on that contract; their parameters,
#: tags and clone are covered above and their Pipeline fit by
#: test_grid_search_over_a_pipeline_fits.
FLOAT32_ONLY = {"MinMaxScaler", "StandardScaler"}

#: Checks that conflict with a documented mojolearn contract, for every
#: estimator. Kept short; each entry says which contract.
EXPECTED_FAILED_CHECKS = {
    # NUMPY-FREE INPUT CONTRACT (NUMPY_FREE_CONTRACT.md): X crosses into the
    # kernels through the buffer protocol or nested Python sequences of
    # numbers. An object-dtype array exports no numeric buffer, and
    # scikit-learn's _NotAnArray exposes only __array__ (no buffer, no len).
    "check_dtype_object": "object-dtype X exports no numeric buffer (NumPy-free input contract)",
    "check_estimators_data_not_an_array": "an __array__-only object is not a buffer or a sequence "
                                          "(NumPy-free input contract)",
    "check_classifier_data_not_an_array": "an __array__-only object is not a buffer or a sequence "
                                          "(NumPy-free input contract)",
    "check_regressor_data_not_an_array": "an __array__-only object is not a buffer or a sequence "
                                         "(NumPy-free input contract)",
    "check_transformer_data_not_an_array": "an __array__-only object is not a buffer or a sequence "
                                           "(NumPy-free input contract)",
    "check_sample_weights_not_an_array": "an __array__-only object is not a buffer or a sequence "
                                         "(NumPy-free input contract)",
    # FLOAT32 ARITHMETIC: fits compute in float32 (float64 X is cast once),
    # and these checks compare weighted and row-repeated fits at float64's
    # default tolerance (assert_allclose, rtol=1e-7).
    "check_sample_weight_equivalence_on_dense_data": "float32 fits compared at float64 tolerance",
    "check_sample_weight_equivalence_on_sparse_data": "float32 fits compared at float64 tolerance",
}


def _expected_failed_checks(estimator):
    return dict(EXPECTED_FAILED_CHECKS)


def _checked_estimators():
    out = []
    for name in SHARED_NAMES:
        if name in FLOAT32_ONLY:
            continue
        try:
            out.append(_make(name))
        except Exception:  # reported by test_params_clone_and_tags
            continue
    return out


if RUN_FITS:
    from sklearn.utils.estimator_checks import parametrize_with_checks

    @parametrize_with_checks(_checked_estimators(), expected_failed_checks=_expected_failed_checks)
    def test_scikit_learn_estimator_checks(estimator, check):
        check(estimator)
else:
    @pytest.mark.skip(reason=FIT_REASON)
    def test_scikit_learn_estimator_checks():
        pass
