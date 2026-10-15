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
    # scikit-learn's SparseCoder check dictionary: 5 atoms over 3 features
    sparse_dictionary = rng.normal(size=(5, 3)).astype(np.float32)
    series = rng.random(42, dtype=np.float32)
    series2d = rng.random((42, 2), dtype=np.float32)
    # The meta-estimators and the classes whose defaults need more rows or
    # features than the checks' small data take scikit-learn's own test
    # parameters (sklearn/utils/_test_common/instance_generator.py,
    # INIT_PARAMS): the same instances scikit-learn checks itself with.
    def trees(cls):
        return [("est1", cls(max_depth=3, random_state=0)), ("est2", cls(max_depth=3, random_state=1))]
    return {
        "MultiOutputClassifier": lambda: ml.MultiOutputClassifier(ml.LogisticRegression(C=1)),
        "MultiOutputRegressor": lambda: ml.MultiOutputRegressor(ml.Ridge()),
        "OneVsRestClassifier": lambda: ml.OneVsRestClassifier(ml.LogisticRegression(C=1)),
        "RFE": lambda: ml.RFE(ml.LogisticRegression(C=1)),
        "CalibratedClassifierCV": lambda: ml.CalibratedClassifierCV(ml.LogisticRegression(C=1), cv=3),
        "SparseCoder": lambda: ml.SparseCoder(sparse_dictionary, transform_n_nonzero_coefs=2),
        "StackingClassifier": lambda: ml.StackingClassifier(trees(ml.DecisionTreeClassifier), cv=3),
        "StackingRegressor": lambda: ml.StackingRegressor(trees(ml.DecisionTreeRegressor), cv=3),
        "VotingClassifier": lambda: ml.VotingClassifier(trees(ml.DecisionTreeClassifier)),
        "VotingRegressor": lambda: ml.VotingRegressor(trees(ml.DecisionTreeRegressor)),
        "CCA": lambda: ml.CCA(n_components=1, max_iter=5),
        "PLSCanonical": lambda: ml.PLSCanonical(n_components=1, max_iter=5),
        "PLSRegression": lambda: ml.PLSRegression(n_components=1, max_iter=5),
        "GaussianRandomProjection": lambda: ml.GaussianRandomProjection(n_components=2),
        "SparseRandomProjection": lambda: ml.SparseRandomProjection(n_components=2),
        "SelectKBest": lambda: ml.SelectKBest(k=1),
        "TSNE": lambda: ml.TSNE(perplexity=2),
        "TruncatedSVD": lambda: ml.TruncatedSVD(n_components=1, n_iter=5),
        "MeanShift": lambda: ml.MeanShift(bandwidth=1.0, max_iter=5),
        "HuberRegressor": lambda: ml.HuberRegressor(max_iter=5),
        # scikit-learn's INIT_PARAMS for both: five coordinate-descent passes
        # per alpha, so a weight-w row and w repeated rows run the same
        # passes (with the default tol=1e-4 stop, float32 rounding can end
        # one fit a pass before the other: LassoCV predict gap 1.4e-2)
        "LassoCV": lambda: ml.LassoCV(max_iter=5, cv=3),
        "ElasticNetCV": lambda: ml.ElasticNetCV(max_iter=5, cv=3),
        "HDBSCAN": lambda: ml.HDBSCAN(min_samples=1),
        # scikit-learn's own test settings for these two (its
        # _test_common/instance_generator.py INIT_PARAMS): with the default
        # 'omp' transform, or NMF's default 200 iterations, fit_transform and
        # transform answer differently by design and its
        # check_transformer_general fails for scikit-learn too.
        "DictionaryLearning": lambda: ml.DictionaryLearning(max_iter=20, transform_algorithm="lasso_lars"),
        "NMF": lambda: ml.NMF(max_iter=500),
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
}

_WEQ = ("check_sample_weight_equivalence_on_dense_data", "check_sample_weight_equivalence_on_sparse_data")

#: scikit-learn's own expected failures of its weight-equivalence checks
#: (sklearn/utils/_test_common/instance_generator.py, PER_ESTIMATOR_XFAIL_CHECKS),
#: for the estimators of the same name, with scikit-learn's reason: bootstrap
#: or random row subsampling, random restarts and shuffled online updates make
#: a weight-w row differ from w repeated rows by design.
SKLEARN_WEIGHT_EQUIVALENCE_XFAIL = (
    "AdaBoostClassifier", "AdaBoostRegressor", "BaggingClassifier", "BaggingRegressor", "BisectingKMeans",
    "GradientBoostingClassifier", "GradientBoostingRegressor", "KMeans", "LinearSVC", "LinearSVR",
    "LogisticRegression", "MiniBatchKMeans", "OneClassSVM", "Perceptron", "RandomForestClassifier",
    "RandomForestRegressor", "SGDClassifier", "SGDOneClassSVM", "SGDRegressor", "SVC", "SVR",
)
_SKLEARN_WEQ_REASON = "sample_weight is not equivalent to removing/repeating samples."


#: Per estimator: the checks scikit-learn's own test suite expects to fail
#: for the same estimator (sklearn/utils/_test_common/instance_generator.py,
#: PER_ESTIMATOR_XFAIL_CHECKS), with scikit-learn's reason.
PER_ESTIMATOR_EXPECTED_FAILED_CHECKS = {
    # A direct Cholesky solve has no iteration count; scikit-learn's
    # RidgeClassifier reports n_iter_ = None for it and xfails the check.
    "RidgeClassifier": {"check_non_transformer_estimators_n_iter": "n_iter_ cannot be easily accessed "
                                                                   "(scikit-learn xfails it too)"},
    # scikit-learn's MultiOutputClassifier refuses a 1-D y, predicts an
    # (n_samples, n_outputs) array and returns predict_proba as a list per
    # output, by design; these four checks assume a single-output classifier
    # and fail on scikit-learn's own MultiOutputClassifier, which its test
    # suite therefore skips (MultiOutputClassifier.__sklearn_tags__:
    # tags._skip_test = True). Ours follows the same contract.
    "MultiOutputClassifier": {
        "check_classifiers_classes": "fits a 1-D y; a 1-D y is refused by design "
                                     "(scikit-learn skips its own MultiOutputClassifier)",
        "check_classifiers_one_label_sample_weights": "fits a 1-D y; a 1-D y is refused by design "
                                                      "(scikit-learn skips its own MultiOutputClassifier)",
        "check_classifiers_train": "expects predict of shape (n_samples,); multi-output predict is "
                                   "(n_samples, n_outputs) (scikit-learn skips its own MultiOutputClassifier)",
        "check_methods_sample_order_invariance": "row-indexes predict_proba; it is a list per output "
                                                 "(scikit-learn skips its own MultiOutputClassifier)",
    },
}

#: Meta-estimators whose weight-equivalence check rests on a member that
#: scikit-learn itself xfails (SKLEARN_WEIGHT_EQUIVALENCE_XFAIL): sample_weight
#: goes to that member's fit, so the wrapper is equivalent exactly when the
#: member is.
WEIGHT_EQUIVALENCE_INHERITED = {
    # members LogisticRegression(C=1), one per output (_factories); scikit-learn
    # also skips every check of its own MultiOutputClassifier
    "MultiOutputClassifier": "its LogisticRegression members' sample_weight is not equivalent to "
                             "removing/repeating samples (scikit-learn xfails LogisticRegression too)",
    # final_estimator LogisticRegression() fits the stacked predictions with sample_weight
    "StackingClassifier": "its final LogisticRegression's sample_weight is not equivalent to "
                          "removing/repeating samples (scikit-learn xfails LogisticRegression too)",
}

#: FLOAT32 ARITHMETIC, per estimator: the weighted fit is scikit-learn's
#: weighted problem exactly (a weight-w row counts as w rows, a weight-0 row
#: as none: the same counts, the same structure), but it runs in float32
#: (float64 X is cast once) and the check compares the weighted and the
#: row-repeated fit at float64's default tolerance (assert_allclose,
#: rtol=1e-7, atol=1e-9). Each entry states the gap measured on NVIDIA
#: (scikit-learn's check data: 15 rows x 30 features, integer weights 0..4).
WEIGHT_EQUIVALENCE_FLOAT32 = {
    # weighted counts sum(w x) per class and feature (x_prep/blocked.mojo
    # csb1_part, w x per row) vs the repeated rows' x + x + ...; the float32
    # joint log-likelihood over 30 features is about -50, whose float32 ulp
    # (3.8e-6) sets the gap of the normalized probabilities
    "MultinomialNB": "float32 fit: predict_proba max relative gap 3.9e-6 (0.34514019 vs 0.34513885), "
                     "4 of 45 entries; same counts (float32 log-likelihood of magnitude ~50)",
    "ComplementNB": "float32 fit: predict_proba max relative gap 3.8e-6 (0.46678317 vs 0.46678495), "
                    "17 of 45 entries; same counts (float32 log-likelihood of magnitude ~50)",
    # weighted means over sum(w), centering, sqrt(w) row scaling, then the
    # solve (linear_model.py, x_linear/ridge.mojo, x_linear/bayes.mojo):
    # scikit-learn's _preprocess_data / _rescale_data construction. With 15
    # rows and 30 features the fits interpolate y in {0, 1, 2}, so entries
    # near 0 are decided by atol=1e-9.
    "LinearRegression": "float32 fit: predict gap 1.8e-7 relative at 1.0 (0.99999982 vs 1.0), "
                        "2.2e-6 absolute near 0 (-3.2e-6 vs -1.0e-6); same weighted problem",
    "Ridge": "float32 fit: predict max relative gap 4.3e-7 (0.27755547 vs 0.27755535); same weighted problem",
    "RidgeClassifier": "float32 fit: decision_function relative gap 2.6e-7 (0.68100601 vs 0.68100584); "
                       "same weighted problem and labels",
    "BayesianRidge": "float32 fit: predict gap 7.7e-7 at 1.0 (0.99999923 vs 1.0), 2.4e-7 absolute near 0; "
                     "same weighted problem (sum(w) in the alpha update)",
    # sqrt(w)-scaled kernel and targets, dual times sqrt(w) (scikit-learn's
    # _solve_cholesky_kernel); a float32 Cholesky of a weighted 15 x 15 kernel
    # with condition ~2e2 sets the gap
    "KernelRidge": "float32 fit: predict max relative gap 4.4e-6 (0.25874203 vs 0.25874317); "
                   "same weighted problem (float32 Cholesky, condition ~2e2)",
    # deviance divided by sum(w); one float32 ulp apart (x_linear/glm.mojo)
    "PoissonRegressor": "float32 fit: predict gap 1 ulp, 1.0e-7 relative (2.31454301 vs 2.31454277); "
                        "same weighted problem",
    "GammaRegressor": "float32 fit: predict gap 1 ulp, 1.0e-7 relative (2.34169173 vs 2.34169197); "
                      "same weighted problem",
    "TweedieRegressor": "float32 fit: predict gap 1 ulp, 1.6e-7 relative (1.50056839 vs 1.50056815); "
                        "same weighted problem",
    # one Ridge per output column (MultiOutputRegressor(Ridge()), _factories)
    "MultiOutputRegressor": "float32 fit of its Ridge members: the gap of Ridge's entry; same weighted problem",
    # weighted means, sqrt(w) row scaling and the weighted held-out errors
    # over the check's LeaveOneGroupOut splits (x_linear/ridgecv.mojo); the
    # same alpha on both fits
    "RidgeCV": "float32 fit: predict max absolute gap 7.2e-7 (3 float32 ulps at 2.0), relative 1.9e-4 only on "
               "an entry near 0; same weighted problem and alpha",
    # weighted coordinate descent over scikit-learn's check setting (max_iter=5,
    # cv=3); the same alpha on both fits (predictions agree to six digits)
    "LassoCV": "float32 fit: predict max absolute gap 1.3e-6 (about 10 float32 ulps at 1-2), relative "
               "1.8e-5 only on an entry near 0.04; same weighted problem and alpha (L40S, 0.8.47)",
    "ElasticNetCV": "float32 fit: the same coordinate descent and gap class as LassoCV's entry; predictions "
                    "agree to six digits, same alpha (L40S, 0.8.47)",
    # final_estimator RidgeCV() over the members' cross-validated predictions
    "StackingRegressor": "float32 fit of its final RidgeCV: predict gap 3.7e-8 absolute, 1.7e-6 relative "
                         "(0.02137140 vs 0.02137136), 3 of 15 entries; same weighted problem",
    # five L-BFGS steps (scikit-learn's max_iter=5 check setting) on the
    # weighted Huber loss (sum of w times the loss; x_linear/huber.mojo)
    "HuberRegressor": "float32 fit, max_iter=5: predict max relative gap 1.2e-5 (0.53079778 vs 0.53080392); "
                      "same weighted problem",
    # loss and penalty scaled by sum(w) of the training rows, held-out
    # accuracy weighted by the raw weights (x_linear/logcv.mojo); the
    # accuracies are exact integer ratios, so both fits pick the same C, and
    # float32 L-BFGS stopping at tol=1e-4 sets the gap
    "LogisticRegressionCV": "float32 L-BFGS fit (tol=1e-4): predict_proba max absolute gap 3.8e-5, max "
                            "relative 4.2e-4, every entry; same weighted problem and C",
}


for _name in SKLEARN_WEIGHT_EQUIVALENCE_XFAIL:
    PER_ESTIMATOR_EXPECTED_FAILED_CHECKS.setdefault(_name, {}).update(
        {check: _SKLEARN_WEQ_REASON + " (scikit-learn xfails it too)" for check in _WEQ})
for _name, _reason in WEIGHT_EQUIVALENCE_INHERITED.items():
    PER_ESTIMATOR_EXPECTED_FAILED_CHECKS.setdefault(_name, {}).update({check: _reason for check in _WEQ})
for _name, _reason in WEIGHT_EQUIVALENCE_FLOAT32.items():
    PER_ESTIMATOR_EXPECTED_FAILED_CHECKS.setdefault(_name, {}).update({check: _reason for check in _WEQ})


def _expected_failed_checks(estimator):
    out = dict(EXPECTED_FAILED_CHECKS)
    out.update(PER_ESTIMATOR_EXPECTED_FAILED_CHECKS.get(type(estimator).__name__, {}))
    return out


def _checked_estimators():
    # MOJOLEARN_SKLEARN_CHECKS_ONLY=Name[,Name]: exact class names, so a runner can
    # give each estimator its own process (tools/sklearn_checks_per_estimator.sh).
    only = os.environ.get("MOJOLEARN_SKLEARN_CHECKS_ONLY")
    only = set(only.split(",")) if only else None
    out = []
    for name in SHARED_NAMES:
        if name in FLOAT32_ONLY or (only is not None and name not in only):
            continue
        try:
            out.append(_make(name))
        except Exception:  # reported by test_params_clone_and_tags
            continue
    return out


#: SparseCoder's dictionary is fixed, so the width of X is set by it. For a
#: check whose X has another width, scikit-learn checks its own SparseCoder
#: with a dictionary of that width (instance_generator.py,
#: PER_ESTIMATOR_CHECK_PARAMS[SparseCoder]: 5 atoms, these widths); these
#: are exactly its entries, no more.
SPARSE_CODER_CHECK_WIDTHS = {
    "check_array_api_input": 10,
    "check_estimators_dtypes": 5,
    "check_dtype_object": 10,
    "check_transformers_unfitted_stateless": 5,
    "check_fit_idempotent": 2,
    "check_transformer_preserve_dtypes": 3,
    "check_set_output_transform": 5,
    "check_global_output_transform_pandas": 5,
    "check_set_output_transform_pandas": 5,
    "check_set_output_transform_polars": 5,
    "check_global_set_output_transform_polars": 5,
    "check_dataframe_column_names_consistency": 8,
    "check_estimators_overwrite_params": 2,
    "check_estimators_fit_returns_self": 2,
    "check_readonly_memmap_input": 2,
    "check_n_features_in_after_fitting": 4,
    "check_fit_check_is_fitted": 2,
    "check_n_features_in": 2,
    "check_positive_only_tag_during_fit": 4,
    "check_fit2d_1sample": 10,
    "check_fit2d_1feature": 1,
}


def _check_function_name(check):
    while hasattr(check, "__wrapped__") or hasattr(check, "func"):
        check = getattr(check, "__wrapped__", None) or check.func
    return check.__name__


def _instance_for_check(estimator, check):
    """scikit-learn's `_yield_instances_for_check` for the estimators it
    gives per-check parameters (SparseCoder)."""
    if type(estimator).__name__ != "SparseCoder":
        return estimator
    width = SPARSE_CODER_CHECK_WIDTHS.get(_check_function_name(check))
    if width is None:
        return estimator
    est = clone(estimator)
    rng = np.random.default_rng(0)
    est.set_params(dictionary=rng.normal(size=(5, width)).astype(np.float32))
    return est


if RUN_FITS:
    from sklearn.utils.estimator_checks import parametrize_with_checks

    @parametrize_with_checks(_checked_estimators(), expected_failed_checks=_expected_failed_checks)
    def test_scikit_learn_estimator_checks(estimator, check):
        check(_instance_for_check(estimator, check))
else:
    @pytest.mark.skip(reason=FIT_REASON)
    def test_scikit_learn_estimator_checks():
        pass
