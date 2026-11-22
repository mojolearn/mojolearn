# SPDX-License-Identifier: Apache-2.0
"""Model-selection and meta-estimator cases run in a subprocess, once per build arm (lanes ma-core, ma-callers).

`python _arm_cases.py OUTDIR CASE [CASE ...]` runs each named case against the Mojolearn installed in THIS
interpreter and writes OUTDIR/<case>.npz (plain arrays, no pickle; an exception is stored as `__error__`, its type
and message). test_proving_model_selection.py runs it with the default wheel and with every `_OFF` arm wheel named
by MOJOLEARN_PROVING_ARM_<DEFINE> (the path of a Python interpreter whose environment has that arm installed), and
compares the files bit for bit. MOJOLEARN_PROVING_DEVICE selects the route: "cpu" (device="cpu", default) or "gpu"
(the default device). Small data, seconds per case. Test tooling only.
"""
import inspect
import os
import sys

import numpy as np

import mojolearn
from mojolearn import model_selection as ms

DEVICE = os.environ.get("MOJOLEARN_PROVING_DEVICE", "cpu")

RNG = np.random.default_rng(0)
X = RNG.random((150, 5), dtype=np.float32)
Y = (X @ np.arange(1, 6, dtype=np.float32)).astype(np.float32)
C = np.searchsorted(np.quantile(Y, [1 / 3, 2 / 3]), Y).astype(np.int32)
B = (C > 0).astype(np.int32)
C4 = np.searchsorted(np.quantile(Y, [0.25, 0.5, 0.75]), Y).astype(np.int32)
X3 = RNG.random((60, 4, 3), dtype=np.float32)
Y3 = X3.sum(axis=(1, 2)).astype(np.float32)
W = (1.0 + RNG.random(150, dtype=np.float32)).astype(np.float32)
X_NAN = X.copy()
X_NAN[7, 1] = np.nan
Y_NAN = Y.copy()
Y_NAN[3] = np.nan


def make(name, **params):
    klass = getattr(mojolearn, name)
    try:
        words = inspect.signature(klass.__init__).parameters
        takes = "device" in words or any(p.kind is p.VAR_KEYWORD for p in words.values())
    except (TypeError, ValueError):
        takes = True
    if DEVICE == "cpu" and takes:
        params["device"] = "cpu"
    return klass(**params)


def bare(name, **params):
    """A Bagging member: its `device` left None, because the members run on the ensemble's device and a member
    with another explicit device is refused in words (BaggingClassifier's `device` docstring); `make` gives the
    ensemble its device."""
    return getattr(mojolearn, name)(**params)


def _cv(result):
    """cross_validate's dict without its clocks."""
    return {k: v for k, v in result.items() if not k.endswith("_time") and k != "estimator"}


CASES = {}


def case(function):
    CASES[function.__name__] = function
    return function


# ---------------------------------------------------------------- ma-core: cross-validation members

@case
def cvs_ridge_default():
    return {"scores": ms.cross_val_score(make("Ridge"), X, Y)}


@case
def cvs_linear_regression_n_jobs():
    return {f"n_jobs_{n}": ms.cross_val_score(make("LinearRegression"), X, Y, cv=4, n_jobs=n) for n in (1, -1)}


@case
def cv_ridge_kfold_and_time_series():
    return {"kfold": _cv(ms.cross_validate(make("Ridge"), X, Y, cv=ms.KFold(n_splits=4, shuffle=False))),
            "time_series": _cv(ms.cross_validate(make("Ridge"), X, Y, cv=ms.TimeSeriesSplit(n_splits=4)))}


@case
def cvs_ridge_neg_mean_squared_error():
    return {"scores": ms.cross_val_score(make("Ridge"), X, Y, scoring="neg_mean_squared_error")}


@case
def cv_logistic_named_scorers():
    return _cv(ms.cross_validate(make("LogisticRegression"), X, C,
                                 scoring=["accuracy", "f1_macro", "neg_log_loss"], return_train_score=True))


@case
def cvs_logistic_f1_labels_1_2():
    return {"scores": ms.cross_val_score(make("LogisticRegression"), X, B + 1, scoring="f1")}


@case
def cvs_logistic_neg_log_loss():
    return {"scores": ms.cross_val_score(make("LogisticRegression"), X, C, scoring="neg_log_loss")}


@case
def cvp_ridge():
    return {"predict": ms.cross_val_predict(make("Ridge"), X, Y)}


@case
def cvp_logistic():
    return {"predict": ms.cross_val_predict(make("LogisticRegression"), X, C),
            "proba": ms.cross_val_predict(make("LogisticRegression"), X, C, method="predict_proba")}


@case
def cvp_svc_decision_function():
    return {"binary": ms.cross_val_predict(make("SVC"), X, B, method="decision_function"),
            "four": ms.cross_val_predict(make("SVC"), X, C4, method="decision_function")}


@case
def cvs_error_score():
    # A number for error_score is cross_validate's (its docstring: "'raise' or float"); cross_val_score documents
    # error_score='raise' only and refuses a number by name.
    return {"scores": ms.cross_validate(make("Ridge", alpha=-1.0), X, Y, error_score=0.5)["test_score"]}


@case
def grid_ridge_and_logistic():
    ridge = ms.GridSearchCV(make("Ridge"), {"alpha": [0.1, 1.0]}).fit(X, Y)
    logistic = ms.GridSearchCV(make("LogisticRegression"), {"C": [0.5, 1.0]}).fit(X, C)
    return {"ridge_predict": ridge.best_estimator_.predict(X),
            "ridge_alpha": ridge.best_estimator_.get_params()["alpha"],
            "logistic_predict": logistic.best_estimator_.predict(X),
            "logistic_classes": logistic.best_estimator_.classes_,
            "logistic_classes_type": type(logistic.best_estimator_.classes_).__name__}


@case
def cvs_member_fallback_families():
    out = {}
    out["pca"] = ms.cross_val_score(make("PCA", n_components=2), X, None)
    out["kmeans"] = ms.cross_val_score(make("KMeans", n_clusters=3, random_state=0), X, None)
    out["logistic"] = ms.cross_val_score(make("LogisticRegression"), X, C)
    out["knn"] = ms.cross_val_score(make("KNeighborsClassifier"), X, C)
    out["nearest_centroid"] = ms.cross_val_score(make("NearestCentroid"), X, C)
    return out


@case
def cvs_neural_and_pls():
    out = {}
    out["mlp_regressor"] = ms.cross_val_score(make("MLPRegressor", max_iter=5, random_state=0), X, Y, cv=3)
    out["mlp_classifier"] = ms.cross_val_score(make("MLPClassifier", max_iter=5, random_state=0), X, C, cv=3)
    out["pls_regression"] = ms.cross_val_score(make("PLSRegression", n_components=2), X, Y, cv=3)
    out["pls_canonical"] = ms.cross_val_score(make("PLSCanonical", n_components=1), X, Y, cv=3)
    out["cca"] = ms.cross_val_score(make("CCA", n_components=1), X, Y, cv=3)
    grid = ms.GridSearchCV(make("MLPRegressor", max_iter=5, random_state=0), {"alpha": [1e-4, 1e-3]}, cv=3)
    out["mlp_grid_predict"] = grid.fit(X, Y).predict(X)
    return out


@case
def cvs_lstm_regressor():
    # cross_val_score takes a 2-D X ("X : array-like of shape (n_samples, n_features)", its docstring); the
    # (n, T, F) sequences an LSTMRegressor reads are refused in words, the same on every arm.
    return {"refusal": _error(lambda: ms.cross_val_score(
        make("LSTMRegressor", hidden_size=4, max_epochs=2, random_state=0), X3, Y3, cv=3))}


# ---------------------------------------------------------------- ma-callers: meta-estimators over members

@case
def bagging_over_trees():
    out = {}
    regressor = make("BaggingRegressor", estimator=bare("DecisionTreeRegressor", random_state=0), n_estimators=5,
                     max_features=0.6, bootstrap_features=True, oob_score=True,
                     random_state=0).fit(X, Y, sample_weight=W)
    out["regressor_predict"] = regressor.predict(X)
    out["regressor_oob"] = regressor.oob_score_
    classifier = make("BaggingClassifier", estimator=bare("RandomForestClassifier", n_estimators=4, random_state=0),
                      n_estimators=4, max_features=0.6, random_state=0).fit(X, C)
    out["classifier_proba"] = classifier.predict_proba(X)
    rare = C.copy()
    rare[:2] = 3                     # a rare class two rows hold
    out["rare_class_predict"] = make("BaggingClassifier", estimator=bare("DecisionTreeClassifier", random_state=0),
                                     n_estimators=4, random_state=0).fit(X, rare).predict(X)
    return out


def _classifiers():
    return [("logistic", make("LogisticRegression")), ("knn", make("KNeighborsClassifier")),
            ("svc", make("SVC", probability=True))]


@case
def voting_stacking_multioutput_ovr():
    out = {}
    soft = mojolearn.VotingClassifier(_classifiers(), voting="soft").fit(X, C)
    hard = mojolearn.VotingClassifier(_classifiers(), voting="hard").fit(X, C)
    out["soft_proba"], out["hard_predict"] = soft.predict_proba(X), hard.predict(X)
    for n in (None, -1, 1, 2):
        out[f"voting_n_jobs_{n}"] = mojolearn.VotingClassifier(_classifiers(), voting="soft",
                                                               n_jobs=n).fit(X, C).predict_proba(X)
    stack = mojolearn.StackingClassifier(_classifiers(), make("LogisticRegression")).fit(X, C)
    # StackingClassifier has predict and predict_proba (python/mojolearn/_expansion_trees.py); it defines no
    # decision_function.
    out["stack_proba"], out["stack_predict"] = stack.predict_proba(X), stack.predict(X)
    regressors = [("ridge", make("Ridge")), ("ols", make("LinearRegression")), ("knn", make("KNeighborsRegressor"))]
    stack_r = mojolearn.StackingRegressor(regressors, make("Ridge")).fit(X, Y)
    out["stack_r_predict"], out["stack_r_transform"] = stack_r.predict(X), stack_r.transform(X)
    stack_many = mojolearn.StackingRegressor(regressors, make("Ridge"), cv=12).fit(X, Y)     # folds > cores
    out["stack_many_folds"] = stack_many.predict(X)
    two = np.stack([C, B], axis=1).astype(np.int32)
    out["multioutput"] = mojolearn.MultiOutputClassifier(make("LogisticRegression")).fit(X, two).predict(X)
    out["ovr"] = mojolearn.OneVsRestClassifier(make("LogisticRegression")).fit(X, C).predict(X)
    out["calibrated"] = mojolearn.CalibratedClassifierCV(make("LogisticRegression"), cv=3).fit(X, C).predict_proba(X)
    return out


@case
def prefit_labels_not_zero_based():
    labels = np.array([3, 7, 9], dtype=np.int64)[C]
    soft = mojolearn.VotingClassifier(_classifiers(), voting="soft").fit(X, labels)
    stack = mojolearn.StackingClassifier(_classifiers(), make("LogisticRegression")).fit(X, labels)
    return {"voting": soft.predict(X), "voting_classes": soft.classes_, "stacking": stack.predict(X)}


@case
def rfe_members():
    out = {}
    for word, estimator, target in (("logistic_binary", make("LogisticRegression"), B),
                                    ("logistic_multiclass", make("LogisticRegression"), C),
                                    ("tree", make("DecisionTreeClassifier", random_state=0), C),
                                    ("forest", make("RandomForestClassifier", n_estimators=4, random_state=0), C)):
        rfe = mojolearn.RFE(estimator, n_features_to_select=3).fit(X, target)
        out[f"{word}_support"], out[f"{word}_ranking"] = rfe.support_, rfe.ranking_
        out[f"{word}_predict"], out[f"{word}_classes"] = rfe.predict(X), rfe.estimator_.classes_
    return out


@case
def explainers_over_members():
    out = {}
    for word, model in (("ridge", make("Ridge").fit(X, Y)), ("ols", make("LinearRegression").fit(X, Y)),
                        ("forest", make("RandomForestRegressor", n_estimators=4, random_state=0).fit(X, Y))):
        out[f"{word}_kernel"] = mojolearn.KernelExplainer(model, X[:20], random_state=0).shap_values(
            X[:5], nsamples=64)
        out[f"{word}_permutation"] = mojolearn.PermutationExplainer(model, X[:20], random_state=0
                                                                    ).shap_values(X[:5], npermutations=4)
    return out


def _error(call):
    try:
        call()
    except Exception as error:   # the refusal's type and words are the compared value
        return f"{type(error).__name__}: {error}"
    return "no error"


@case
def nan_refusals():
    out = {}
    builders = {
        "voting": lambda: mojolearn.VotingClassifier(_classifiers(), voting="soft"),
        "stacking": lambda: mojolearn.StackingClassifier(_classifiers(), make("LogisticRegression")),
        "multioutput": lambda: mojolearn.MultiOutputRegressor(make("Ridge")),
        "ovr": lambda: mojolearn.OneVsRestClassifier(make("LogisticRegression")),
        "calibrated": lambda: mojolearn.CalibratedClassifierCV(make("LogisticRegression"), cv=3),
        # Bagging members are tree estimators (BaggingRegressor's `estimator` docstring; Ridge is refused by name).
        "bagging": lambda: make("BaggingRegressor", estimator=bare("DecisionTreeRegressor", random_state=0),
                                oob_score=True, random_state=0),
        "rfe": lambda: mojolearn.RFE(make("LogisticRegression"), n_features_to_select=2),
    }
    regression = {"multioutput", "bagging"}
    for word, build in builders.items():
        y = np.stack([Y, Y], axis=1) if word == "multioutput" else (Y if word in regression else C)
        y_bad = np.stack([Y_NAN, Y_NAN], axis=1) if word == "multioutput" else (Y_NAN if word in regression else None)
        out[f"{word}_fit_x"] = _error(lambda: build().fit(X_NAN, y))
        if y_bad is not None:
            out[f"{word}_fit_y"] = _error(lambda: build().fit(X, y_bad))
        fitted = build().fit(X, y)
        out[f"{word}_predict_x"] = _error(lambda: fitted.predict(X_NAN))
    out["cross_val_score"] = _error(lambda: ms.cross_val_score(make("Ridge"), X_NAN, Y))
    out["cross_val_predict"] = _error(lambda: ms.cross_val_predict(make("Ridge"), X_NAN, Y))
    out["grid_refit"] = _error(lambda: ms.GridSearchCV(make("Ridge"), {"alpha": [1.0]}).fit(X_NAN, Y))
    return out


@case
def missing_class_folds():
    ordered = np.sort(C)                       # unshuffled folds: the first fold sees no class 2 in training
    return {
        "stacking": _error(lambda: mojolearn.StackingClassifier(_classifiers(), make("LogisticRegression"),
                                                                cv=ms.KFold(n_splits=3)).fit(X, ordered)),
        "calibrated": _error(lambda: mojolearn.CalibratedClassifierCV(make("LogisticRegression"),
                                                                      cv=ms.KFold(n_splits=3)).fit(X, ordered)),
    }


# ---------------------------------------------------------------- the runner

def _flatten(prefix, value, into):
    if isinstance(value, dict):
        for key, inner in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), inner, into)
        return
    if isinstance(value, str):
        into[prefix] = np.asarray(value)
        return
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], str):
        into[prefix] = np.asarray(value, dtype=str)
        return
    try:
        array = np.asarray(value)
    except Exception:
        array = np.asarray(memoryview(value))
    if array.dtype == object:
        array = array.astype(str)
    into[prefix] = array
    into[prefix + ".__type__"] = np.asarray(type(value).__name__)


def run(outdir, names):
    os.makedirs(outdir, exist_ok=True)
    for name in names:
        arrays = {}
        try:
            _flatten("", CASES[name](), arrays)
        except Exception as error:   # recorded, compared across arms, and failed by the test
            arrays = {"__error__": np.asarray(f"{type(error).__name__}: {error}")}
        np.savez(os.path.join(outdir, name + ".npz"), **arrays)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2:])
