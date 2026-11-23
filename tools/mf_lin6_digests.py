"""Before/after digests of the member routes lane mf-lin6 added (2026-11-23) in the XLinear, LinearSvm,
Discriminant, KernelApprox and Cd families, host column. One line per configuration and output: name and sha256
of the output bytes (dtype and shape included); a refusal is digested by its words. Base:
origin/lane/member-fits-6 4b489b963f (each setting below made the member admission answer False and the class's
own door ran the fit), head lane/mf-lin6 (the member runs it). Test tool, never product runtime.

Every configuration is fitted through GridSearchCV with one given split (train = test = every row), a one-point
grid on the estimator's first parameter by name, refit=True: the refit fits the estimator as a member. Scoring:
'r2' for a regressor, 'accuracy' for a classifier, none for a transformer. Digested: mean_test_score, the refit
estimator's fitted attributes and one output.

Rows covered (mf-cal5-work/fallback-audit.tsv): xlinear 191, 195, 212, 244, 249, 251, 255, 507, 512, 517, 522;
linear_svm 92 (no unit can build an image with -D MOJOLEARN_PY2MOJO_linear_OFF here: checks/native_tests/
mf_lin6_cv.mojo case 6 is its check; `linearsvc_binary` is the default image's control); discriminant 99;
kernel_approx 110; cd 52.

EXPECTED: every line equal, except, stated: the `*_raised` lines keep their words but may now come from the member
admission instead of the door (the same text); `lda_nc0_*`: before, the door fitted n_components=0 (no
scikit-learn check) and every attribute was digested; after, one `lda_nc0_raised` line with scikit-learn's
InvalidParameterError words (a bits/behaviour change, stated); `*_fast` lines where this installation has no fast
kernel methods image (both sides raise the same words). Configurations are independent: one process each across
the host cores.

    python tools/mf_lin6_digests.py > digests.txt
"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import os
import sys

import numpy as np


def _digest(value):
    a = np.asarray(value)
    h = hashlib.sha256()
    h.update(f"{a.dtype.str}{a.shape}".encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:24]


def _data(n=48, d=4, seed=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = (X[:, 0] - 0.5 * X[:, 1]).astype(np.float32)
    return X, y


def _pairs(n, k=3):
    rows = np.arange(n)
    return [(rows[rows % k != f], rows[rows % k == f]) for f in range(k)]


def _config(name):
    """(estimator, X, y, scoring, attribute names, output method) of configuration `name`."""
    import mojolearn
    from mojolearn import model_selection as ms
    X, y = _data()
    labels = (y > 0).astype(np.int64)
    n = len(X)
    if name == "ridgecv_listed":
        return (mojolearn.RidgeCV(cv=_pairs(n)), X, y, "r2", ["coef_", "intercept_", "alpha_", "best_score_"],
                "predict")
    if name == "lassocv_listed":
        return (mojolearn.LassoCV(cv=_pairs(n)), X, y, "r2", ["coef_", "intercept_", "alpha_", "mse_path_"],
                "predict")
    if name == "enetcv_listed":
        return (mojolearn.ElasticNetCV(cv=_pairs(n), l1_ratio=[0.5, 1.0]), X, y, "r2",
                ["coef_", "intercept_", "alpha_", "l1_ratio_", "mse_path_"], "predict")
    if name == "logcv_listed":
        return (mojolearn.LogisticRegressionCV(cv=_pairs(n)), X, labels, "accuracy",
                ["coef_", "intercept_", "C_", "scores_"], "predict")
    if name == "lassocv_listed_tuple":
        return (mojolearn.LassoCV(cv=tuple(_pairs(n))), X, y, "r2", ["coef_", "alpha_"], "predict")
    if name == "lassocv_listed_empty":
        return (mojolearn.LassoCV(cv=[]), X, y, "r2", ["coef_"], "predict")
    if name == "lassocv_listed_three":
        rows = np.arange(n)
        return (mojolearn.LassoCV(cv=[(rows[:30], rows[30:], rows[:5])]), X, y, "r2", ["coef_"], "predict")
    if name == "lassocv_listed_bad_index":
        return (mojolearn.LassoCV(cv=[(np.arange(30).astype(np.float32), np.arange(30, 48))]), X, y, "r2",
                ["coef_"], "predict")
    if name == "lassocv_kfold_one":
        kf = ms.KFold(n_splits=3)
        kf.n_splits = 1
        return (mojolearn.LassoCV(cv=kf), X, y, "r2", ["coef_"], "predict")
    if name == "lassocv_repeated_zero":
        return (mojolearn.LassoCV(cv=ms.RepeatedKFold(n_splits=3, n_repeats=0, random_state=1)), X, y, "r2",
                ["coef_"], "predict")
    if name == "lassocv_timeseries":
        return (mojolearn.LassoCV(cv=ms.TimeSeriesSplit(n_splits=3)), X, y, "r2", ["coef_", "alpha_", "mse_path_"],
                "predict")
    if name == "lda_nc0":
        return (mojolearn.LinearDiscriminantAnalysis(n_components=0), X, labels, "accuracy",
                ["coef_", "intercept_", "scalings_"], "predict")
    if name == "lda_nc1":
        return (mojolearn.LinearDiscriminantAnalysis(n_components=1), X, labels, "accuracy",
                ["coef_", "intercept_", "scalings_"], "transform")
    if name == "nystroem_fast":
        return (mojolearn.Nystroem(kernel="rbf", gamma=0.5, n_components=8, random_state=3, numeric_mode="fast"), X,
                None, None, ["components_", "normalization_"], "transform")
    if name == "rbf_fast":
        return (mojolearn.RBFSampler(gamma=0.5, n_components=8, random_state=3, numeric_mode="fast"), X, None, None,
                ["random_weights_", "random_offset_"], "transform")
    if name == "rbf_scale_fast":
        return (mojolearn.RBFSampler(gamma="scale", n_components=8, random_state=3, numeric_mode="fast"), X, None,
                None, ["random_weights_", "random_offset_"], "transform")
    if name == "hostlasso":
        from mojolearn._classical_host import HostLasso
        return (HostLasso(), X, y, "r2", ["coef_"], "predict")
    if name == "linearsvc_binary":
        return (mojolearn.LinearSVC(), X, labels, "accuracy", ["coef_", "intercept_"], "predict")
    raise ValueError(name)


def _unit(name):
    from mojolearn.model_selection import GridSearchCV
    out = []
    try:
        est, X, y, scoring, attrs, method = _config(name)
        every = np.arange(len(X))
        first = sorted(est.get_params())[0]
        search = GridSearchCV(est, {first: [est.get_params()[first]]}, cv=[(every, every)], scoring=scoring,
                              refit=True)
        search.fit(X, y)
        out.append((name + "_scores", _digest(np.asarray(search.cv_results_["mean_test_score"], dtype=np.float64))))
        best = search.best_estimator_
        for attr in attrs:
            value = getattr(best, attr, None)
            out.append((f"{name}_{attr}", "None" if value is None else _digest(value)))
        try:
            out.append((f"{name}_{method}", _digest(getattr(best, method)(X[:8]))))
        except Exception as e:  # noqa: BLE001 - the words are the digest
            out.append((f"{name}_{method}_raised", f"{type(e).__name__}: {e}"))
    except Exception as e:  # noqa: BLE001 - the words are the digest
        out.append((name + "_raised", f"{type(e).__name__}: {e}"))
    return out


UNITS = ["ridgecv_listed", "lassocv_listed", "enetcv_listed", "logcv_listed", "lassocv_listed_tuple",
         "lassocv_listed_empty", "lassocv_listed_three", "lassocv_listed_bad_index", "lassocv_kfold_one",
         "lassocv_repeated_zero", "lassocv_timeseries", "lda_nc0", "lda_nc1", "nystroem_fast", "rbf_fast",
         "rbf_scale_fast", "hostlasso", "linearsvc_binary"]


def main():
    with ProcessPoolExecutor(max_workers=min(len(UNITS), os.cpu_count() or 1)) as pool:
        for lines in pool.map(_unit, UNITS):
            for name, value in lines:
                print(name, value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
