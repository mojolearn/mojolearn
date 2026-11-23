#!/usr/bin/env python3
"""Output digests of the SHAP model classes lane mf-shap2 moved onto native members (2026-11-23), extending
tools/mf_shaprfe_digests.py (its configurations are kept, under the same names). A testing tool, never product
execution.

(C, mf-shap2) KernelExplainer and PermutationExplainer over the predict-only classifiers (LinearSVC,
RidgeClassifier, PassiveAggressiveClassifier, Perceptron, SGDClassifier(loss='hinge'), SVC(probability=False),
NearestCentroid), now through `AgnMember`'s label route (member codes mapped to `classes_` by the glue gather),
with labels 0..k-1, with shifted float labels and with string labels (the door's refusal); CategoricalNB (now
NaiveBayes member kind 4) explained, fitted, predicted, inside BaggingClassifier and VotingClassifier, and under
RFE; VotingClassifier(voting='soft') and VotingRegressor over native members (now `AgnVote`), a vote with a tree
member and a hard vote (both kept on their old paths).

(A) KernelExplainer and PermutationExplainer over the classes that left the door's `_DoorModel`: a member whose
predict_proba is float64 (LogisticRegression, SVC(probability=True), DARTClassifier,
GaussianProcessClassifier, GradientBoostingClassifier), now through the native handle `AgnMember` with the RUN's
float32 narrowing, and the naive Bayes / discriminant members, which now run with the GIL released
(MEMBER_GIL_DOOR). Each class x {kernel identity, kernel logit (binary), permutation}, plus a multi-class and a
many-rows (parallel chunks) case.
(B) RFE over LinearDiscriminantAnalysis (svd, lsqr: now through the discriminant member's coef_ export),
QuadraticDiscriminantAnalysis, GaussianNB and MultinomialNB (now refused before any round is fitted) x step
{1, 2} x n_features_to_select {None, 2}: ranking_, support_ and the final estimator's predict_proba on 10 rows.

One line per configuration, `<name> sha256=<digest of the outputs' and fitted state's bytes>` or
`<name> error <class>: <first line of the message>` (a refusal is digested by its words). Run it on two builds of
the CPU column, base origin/lane/member-fits d3a8864541 and lane/mf-shap2's head, and diff the files: equal lines
mean equal bits.

    PYTHONPATH=<repo>/python python3 tools/mf_shap2_digests.py > out.txt

Expected: every line equal. The only lines that may differ are `rfe_categorical_nb_*`: CategoricalNB is now a
member of the naive Bayes family, which RFE refuses before any round (the door's 'auto' words; scikit-learn's
attrgetter words for a named getter), where its own-door path fitted every round's clone first. Any other
difference means stop and send it back.

Parallel plan: the configurations are independent, one per worker process (at most the allocated cores, at most
the configuration count).
"""
import hashlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor


def _data(kind, n=300, d=5, seed=0, positive=False):
    import numpy as np
    rng = np.random.default_rng(seed)
    if kind.startswith("cat"):
        # CategoricalNB: integer categories 0..3 per feature
        X = rng.integers(0, 4, (n, d)).astype(np.float32)
        k = int(kind[3:])
        y = ((X[:, 0] + X[:, 1]).astype(np.int64) % k).astype(np.int64)
        return X, y
    if kind == "reg":
        X = rng.normal(size=(n, d)).astype(np.float32)
        y = (X[:, 0] - 0.5 * X[:, 1] + 0.1 * X[:, 2]).astype(np.float32)
        return X, y
    X = rng.normal(size=(n, d)).astype(np.float32)
    k = int(kind[3:])
    y = (np.digitize(X[:, 0] + 0.5 * X[:, 1], np.linspace(-1, 1, k - 1))).astype(np.int64)
    if positive:
        X = np.abs(X)
    return X, y


def _labels(y, how):
    """The class labels a configuration fits with: the codes, shifted floats, or strings."""
    import numpy as np
    if how == "float":
        return y.astype(np.float64) * 2.5 - 1.0
    if how == "str":
        return np.array(["c%d" % v for v in y])  # glue: the fixture's label strings
    return y


_SHAP_MODELS = {
    "logistic": ("LogisticRegression", {}, False),
    "svc_proba": ("SVC", dict(probability=True, random_state=0), False),
    "dart": ("DARTClassifier", dict(n_estimators=12, random_state=0), False),
    "gpc": ("GaussianProcessClassifier", dict(random_state=0), False),
    "gbdt_clf": ("GradientBoostingClassifier", dict(n_estimators=12, random_state=0), False),
    "gaussian_nb": ("GaussianNB", {}, False),
    "multinomial_nb": ("MultinomialNB", {}, True),
    "lda": ("LinearDiscriminantAnalysis", {}, False),
    "qda": ("QuadraticDiscriminantAnalysis", {}, False),
}

_RFE_MODELS = {
    "lda_svd": ("LinearDiscriminantAnalysis", dict(solver="svd"), False),
    "lda_lsqr": ("LinearDiscriminantAnalysis", dict(solver="lsqr"), False),
    "qda": ("QuadraticDiscriminantAnalysis", {}, False),
    "gaussian_nb": ("GaussianNB", {}, False),
    "multinomial_nb": ("MultinomialNB", {}, True),
}


# (C) predict-only classifiers, explained through predict (the label route)
_LABEL_MODELS = {
    "linear_svc": ("LinearSVC", dict(random_state=0)),
    "ridge_clf": ("RidgeClassifier", {}),
    "passive_aggressive": ("PassiveAggressiveClassifier", dict(random_state=0)),
    "perceptron": ("Perceptron", dict(random_state=0)),
    "sgd_hinge": ("SGDClassifier", dict(loss="hinge", random_state=0)),
    "svc_noproba": ("SVC", dict(probability=False)),
    "nearest_centroid": ("NearestCentroid", {}),
}

# (C) votes: (class, members as (name, class, kwargs), vote kwargs, data kind)
_VOTES = {
    "vote_soft": ("VotingClassifier", (("lr", "LogisticRegression", {}), ("nb", "GaussianNB", {})),
                  dict(voting="soft", weights=[1.0, 2.0]), "clf3"),
    "vote_soft_tree": ("VotingClassifier", (("lr", "LogisticRegression", {}),
                                            ("dt", "DecisionTreeClassifier", dict(max_depth=3, random_state=0))),
                       dict(voting="soft"), "clf3"),
    "vote_hard": ("VotingClassifier", (("lr", "LogisticRegression", {}), ("nb", "GaussianNB", {})),
                  dict(voting="hard"), "clf3"),
    "vote_regress": ("VotingRegressor", (("lin", "LinearRegression", {}), ("ridge", "Ridge", {})),
                     dict(weights=[2.0, 1.0]), "reg"),
}


def _cases():
    cases = {}
    for key, (cls, mkw) in _LABEL_MODELS.items():  # glue: the fixed configuration table
        for how in ("codes", "float", "str"):  # glue: the fixed configuration table
            cases["shap_k_label_" + key + "_" + how] = ("K", cls, mkw, False, "clf3", {}, {}, 8, how)
            cases["shap_p_label_" + key + "_" + how] = ("P", cls, mkw, False, "clf3", dict(random_state=0),
                                                        dict(npermutations=20), 8, how)
    cases["shap_k_label_linear_svc_many_rows"] = ("K", "LinearSVC", dict(random_state=0), False, "clf2", {}, {}, 97,
                                                  "codes")
    for kind in ("cat2", "cat3"):  # glue: the fixed configuration table
        cases["shap_k_categorical_nb_" + kind] = ("K", "CategoricalNB", {}, False, kind, {}, {}, 8, "codes")
        cases["shap_p_categorical_nb_" + kind] = ("P", "CategoricalNB", {}, False, kind, dict(random_state=0),
                                                  dict(npermutations=20), 8, "codes")
        cases["fit_categorical_nb_" + kind] = ("F", "CategoricalNB", {}, False, kind, {}, {}, 0, "codes")
        cases["bag_categorical_nb_" + kind] = ("B", "CategoricalNB", {}, False, kind, {}, {}, 0, "codes")
        cases["vote_categorical_nb_" + kind] = ("V", "CategoricalNB", {}, False, kind, {}, {}, 0, "codes")
    cases["fit_categorical_nb_min_categories"] = ("F", "CategoricalNB", dict(min_categories=[2, 7, 5, 4, 6]), False,
                                                  "cat3", {}, {}, 0, "codes")
    cases["shap_k_categorical_nb_out_of_range"] = ("K", "CategoricalNB", {}, False, "cat3", {}, {}, -8, "codes")
    for key in _VOTES:  # glue: the fixed configuration table
        cases["shap_k_" + key] = ("KV", key, {}, False, "", {}, {}, 8, "codes")
        cases["shap_p_" + key] = ("PV", key, {}, False, "", dict(random_state=0), dict(npermutations=20), 8, "codes")
    cases["shap_k_vote_soft_many_rows"] = ("KV", "vote_soft", {}, False, "", {}, {}, 97, "codes")
    for step in (1, 2):  # glue: the fixed configuration table
        cases["rfe_categorical_nb_step%d" % step] = ("R", "CategoricalNB", {}, False, "cat3",
                                                     dict(step=step, n_features_to_select=2), {}, 0, "codes")
    cases["rfe_categorical_nb_getter_coef"] = ("R", "CategoricalNB", {}, False, "cat3",
                                               dict(step=1, n_features_to_select=2, importance_getter="coef_"), {},
                                               0, "codes")
    for key, (cls, mkw, positive) in _SHAP_MODELS.items():  # glue: the fixed configuration table
        cases["shap_k_" + key + "_identity"] = ("K", cls, mkw, positive, "clf2", {}, {}, 8, "codes")
        cases["shap_k_" + key + "_logit"] = ("K", cls, mkw, positive, "clf2", dict(link="logit"), {}, 8, "codes")
        cases["shap_p_" + key] = ("P", cls, mkw, positive, "clf2", dict(random_state=0), dict(npermutations=20), 8,
                                  "codes")
    cases["shap_k_logistic_clf3"] = ("K", "LogisticRegression", {}, False, "clf3", {}, {}, 8, "codes")
    cases["shap_p_logistic_many_rows"] = ("P", "LogisticRegression", {}, False, "clf2", dict(random_state=1),
                                          dict(npermutations=10), 97, "codes")
    cases["shap_k_dart_many_rows"] = ("K", "DARTClassifier", dict(n_estimators=12, random_state=0), False, "clf2",
                                      {}, {}, 97, "codes")
    for key, (cls, mkw, positive) in _RFE_MODELS.items():  # glue: the fixed configuration table
        for step in (1, 2):  # glue: the fixed configuration table
            for nsel in (None, 2):  # glue: the fixed configuration table
                name = "rfe_%s_step%d_n%s" % (key, step, "none" if nsel is None else nsel)
                cases[name] = ("R", cls, mkw, positive, "clf3", dict(step=step, n_features_to_select=nsel), {}, 0,
                               "codes")
        cases["rfe_" + key + "_getter_coef"] = ("R", cls, mkw, positive, "clf3",
                                                dict(step=1, n_features_to_select=2, importance_getter="coef_"),
                                                {}, 0, "codes")
    return cases


def _digest(*arrays):
    import numpy as np
    h = hashlib.sha256()
    for a in arrays:  # glue: the configuration's few output arrays
        a = np.asarray(a)
        h.update(str(a.shape).encode())
        h.update(str(a.dtype).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _vote(key, ml):
    cls, members, vkw, kind = _VOTES[key]
    estimators = [(name, getattr(ml, c)(**kw)) for name, c, kw in members]  # glue: the fixed member table
    return getattr(ml, cls)(estimators, **vkw), kind


def _run(name):
    import numpy as np
    import mojolearn as ml
    which, cls, mkw, positive, kind, ekw, vkw, rows, how = _cases()[name]
    try:
        if which in ("KV", "PV"):
            model, kind = _vote(cls, ml)
            X, y = _data(kind)
            model.fit(X, y)
            which = which[0]
        else:
            X, y = _data(kind, positive=positive)
            y = _labels(y, how)
            if which == "R":
                r = ml.RFE(getattr(ml, cls)(**mkw), **ekw).fit(X, y)
                proba = r.predict_proba(X[:10])
                return f"{name} sha256={_digest(r.ranking_, np.asarray(r.support_), proba)}"
            if which == "F":
                m = getattr(ml, cls)(**mkw).fit(X, y)
                return f"{name} sha256={_digest(m.predict_proba(X[:50]), m.predict(X[:50]), m._flp, m._cc)}"
            if which == "B":
                m = ml.BaggingClassifier(getattr(ml, cls)(**mkw), n_estimators=4, random_state=0).fit(X, y)
                return f"{name} sha256={_digest(m.predict_proba(X[:50]), m.predict(X[:50]))}"
            if which == "V":
                m = ml.VotingClassifier([("cat", getattr(ml, cls)(**mkw)), ("lr", ml.LogisticRegression())],
                                        voting="soft").fit(X, y)
                return f"{name} sha256={_digest(m.predict_proba(X[:50]), m.predict(X[:50]))}"
            model = getattr(ml, cls)(**mkw).fit(X, y)
        explainer = (ml.KernelExplainer if which == "K" else ml.PermutationExplainer)(model, X[:20], **ekw)
        if rows < 0:
            # a category index above the fitted range in the explained rows: the model's own refusal
            rows_x = X[200:200 - rows].copy()
            rows_x[0, 0] = 9.0
        else:
            rows_x = X[200:200 + rows].copy()
        phi = np.asarray(explainer.shap_values(rows_x, **vkw))
        ev = np.asarray(explainer.expected_value, dtype=np.float64)
        return f"{name} sha256={_digest(phi, ev)}"
    except Exception as e:  # noqa: BLE001 - the line records the refusal
        return f"{name} error {type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"


def main():
    names = sorted(_cases())
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:
        cores = os.cpu_count() or 1
    workers = max(1, min(int(os.environ.get("MF_SHAP2_WORKERS", cores)), len(names)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for line in pool.map(_run, names):
            print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
