# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# ONE LANE PER PUBLIC MODEL-SELECTION TOOL (mojolearn.cross_val_score and the
# mojolearn.model_selection functions), sub-lane vdv-cov-func, 2026-11-24.
#
# Bounded: 300 rows, 3 folds, small linear members. Every member is built
# through `ml` inside the lane, so the harness's device-bound view gives it the
# running side (members fitted on the side). Each lane hands its calls to
# `_functions_resident` (tools/identity_lanes/functions.py) with the arena form:
# on the GPU side the same calls run twice inside a `resident()` scope on the
# same X and y, must give the same bytes, and must read the device store (fold
# rows selected on the device, members fit and scored there).


def _functions_msel_fx(X, n=300):
    """The model-selection fixture: X (n, 6), a regression target, 3-class
    labels and 4 groups, built once per lane."""
    Xs = _functions_mat(X, n, 6, "fms:x")
    y = np.ascontiguousarray((Xs[:, 0] * np.float32(0.5) + Xs[:, 3] * np.float32(0.25)
                              + _hw((n,), "fms:noise", -0.1, 0.1)).astype(np.float32))
    c = _functions_classes(X, n, 3, "fms:c")
    g = np.ascontiguousarray((np.arange(n) % 4).astype(np.int64))
    return dict(X=Xs, y=y, c=c, g=g)


def _functions_msel_splits(splits):
    """(train, test) pairs as one int64 stream: -1 between train and test,
    -2 after each pair."""
    out = []
    for tr, te in splits:  # glue: the folds, in order
        out.extend(np.asarray(tr, dtype=np.int64).tolist())
        out.append(-1)
        out.extend(np.asarray(te, dtype=np.int64).tolist())
        out.append(-2)
    return np.asarray(out, dtype=np.int64)


@lane("fn-msel-cross-val-score")
def _(ml, X, yc, yr, Xh=None):
    """cross_val_score (as mojolearn.cross_val_score and
    model_selection.cross_val_score): Ridge on 3 folds, default and named scorer."""
    f = _functions_msel_fx(X)
    return _functions_resident(ml, "cross_val_score", lambda p: dict(
        default=_functions_d(ml.cross_val_score(ml.Ridge(alpha=1.0), p(f["X"]), p(f["y"]), cv=3)),
        mae=_functions_d(ml.model_selection.cross_val_score(ml.Ridge(alpha=1.0), p(f["X"]), p(f["y"]), cv=3,
                                                            scoring="neg_mean_absolute_error")),
        clf=_functions_d(ml.cross_val_score(ml.LogisticRegression(max_iter=50), p(f["X"]), p(f["c"]), cv=3))))


@lane("fn-msel-cross-validate")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_msel_fx(X)

    def scores(r):
        return {k: v for k, v in r.items() if k.startswith(("test_", "train_"))}  # glue: the score arrays only

    return _functions_resident(ml, "model_selection.cross_validate", lambda p: dict(
        multi=_functions_d(scores(ml.model_selection.cross_validate(
            ml.Ridge(alpha=1.0), p(f["X"]), p(f["y"]), cv=3, scoring=["r2", "neg_mean_squared_error"],
            return_train_score=True)))))


@lane("fn-msel-cross-val-predict")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_msel_fx(X)
    return _functions_resident(ml, "model_selection.cross_val_predict", lambda p: dict(
        predict=_functions_d(ml.model_selection.cross_val_predict(ml.Ridge(alpha=1.0), p(f["X"]), p(f["y"]), cv=3)),
        proba=_functions_d(ml.model_selection.cross_val_predict(ml.LogisticRegression(max_iter=50), p(f["X"]),
                                                                p(f["c"]), cv=3, method="predict_proba"))))


@lane("fn-msel-learning-curve")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_msel_fx(X)
    return _functions_resident(ml, "model_selection.learning_curve", lambda p: dict(
        curve=_functions_d(ml.model_selection.learning_curve(
            ml.Ridge(alpha=1.0), p(f["X"]), p(f["y"]), train_sizes=[0.5, 1.0], cv=3,
            scoring="neg_mean_squared_error", error_score="raise")),
        shuffled=_functions_d(ml.model_selection.learning_curve(
            ml.Ridge(alpha=1.0), p(f["X"]), p(f["y"]), train_sizes=[0.4, 0.8], cv=3, shuffle=True,
            random_state=5, error_score="raise"))))


@lane("fn-msel-validation-curve")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_msel_fx(X)
    return _functions_resident(ml, "model_selection.validation_curve", lambda p: dict(
        curve=_functions_d(ml.model_selection.validation_curve(
            ml.Ridge(), p(f["X"]), p(f["y"]), param_name="alpha", param_range=[0.5, 2.0, 8.0], cv=3,
            scoring="neg_mean_squared_error", error_score="raise"))))


@lane("fn-msel-permutation-test-score")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_msel_fx(X)
    return _functions_resident(ml, "model_selection.permutation_test_score", lambda p: dict(
        plain=_functions_d(ml.model_selection.permutation_test_score(
            ml.Ridge(alpha=1.0), p(f["X"]), p(f["y"]), cv=3, n_permutations=4, random_state=3)),
        groups=_functions_d(ml.model_selection.permutation_test_score(
            ml.Ridge(alpha=1.0), p(f["X"]), p(f["y"]), groups=p(f["g"]), cv=3, n_permutations=4, random_state=3))))


@lane("fn-msel-train-test-split")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_msel_fx(X)
    return _functions_resident(ml, "model_selection.train_test_split", lambda p: dict(
        plain=_functions_d(ml.model_selection.train_test_split(p(f["X"]), p(f["y"]), test_size=0.25, random_state=13)),
        stratified=_functions_d(ml.model_selection.train_test_split(p(f["X"]), p(f["c"]), test_size=0.3,
                                                                    random_state=13, stratify=p(f["c"])))))


@lane("fn-msel-check-cv")
def _(ml, X, yc, yr, Xh=None):
    """check_cv's splitter for a classifier (stratified) and a regressor, and
    its shuffled form: the folds it hands out."""
    f = _functions_msel_fx(X)
    ms = ml.model_selection
    return _functions_resident(ml, "model_selection.check_cv", lambda p: dict(
        clf=_functions_d(_functions_msel_splits(ms.check_cv(3, p(f["c"]), classifier=True).split(p(f["X"]), p(f["c"])))),
        reg=_functions_d(_functions_msel_splits(ms.check_cv(4, p(f["y"])).split(p(f["X"]), p(f["y"])))),
        shuffled=_functions_d(_functions_msel_splits(ms.check_cv(3, p(f["c"]), classifier=True, shuffle=True,
                                                                  random_state=7).split(p(f["X"]), p(f["c"]))))))


@lane("fn-msel-default-folds")
def _(ml, X, yc, yr, Xh=None):
    f = _functions_msel_fx(X)
    ms = ml.model_selection
    return _functions_resident(ml, "model_selection.default_folds", lambda p: dict(
        strat=_functions_d(_functions_msel_splits(ms.default_folds(p(f["c"]), 3, classifier=True))),
        kfold=_functions_d(_functions_msel_splits(ms.default_folds(p(f["y"]), 5)))))


@lane("fn-msel-split-descriptor")
def _(ml, X, yc, yr, Xh=None):
    """split_descriptor: the fold metadata model selection derives for a
    classifier on 3 and 5 folds, as its key=value bytes in key order."""
    f = _functions_msel_fx(X)
    ms = ml.model_selection

    def text(d):
        return np.frombuffer("|".join(f"{k}={d[k]}" for k in sorted(d)).encode("utf-8"), dtype=np.uint8)

    return _functions_resident(ml, "model_selection.split_descriptor", lambda p: dict(
        cv3=_functions_d(text(ms.split_descriptor(p(f["X"]), p(f["c"]), estimator=ml.LogisticRegression(), cv=3))),
        cv5=_functions_d(text(ms.split_descriptor(p(f["X"]), p(f["y"]), estimator=ml.Ridge(), cv=5)))))


@lane("fn-msel-scorers")
def _(ml, X, yc, yr, Xh=None):
    """get_scorer and make_scorer applied to a Ridge fitted on the side: the
    named r2 scorer, a made pinball scorer and a made r2 scorer."""
    f = _functions_msel_fx(X)
    ms = ml.model_selection

    def score(p):
        Xp, yp = p(f["X"]), p(f["y"])
        model = ml.Ridge(alpha=1.0).fit(Xp, yp)
        return dict(named=_functions_d(ms.get_scorer("r2")(model, Xp, yp)),
                    pinball=_functions_d(ms.make_scorer(ml.metrics.mean_pinball_loss, alpha=0.3,
                                                        greater_is_better=False)(model, Xp, yp)),
                    made=_functions_d(ms.make_scorer(ml.metrics.r2_score)(model, Xp, yp)))

    return _functions_resident(ml, "model_selection.get_scorer/make_scorer", lambda p: score(p))


_batch_decl(_FUNCTIONS_BATCH_NA, "fn-msel-cross-val-score", "fn-msel-cross-validate", "fn-msel-cross-val-predict",
            "fn-msel-learning-curve", "fn-msel-validation-curve", "fn-msel-permutation-test-score",
            "fn-msel-train-test-split", "fn-msel-check-cv", "fn-msel-default-folds", "fn-msel-split-descriptor",
            "fn-msel-scorers")
