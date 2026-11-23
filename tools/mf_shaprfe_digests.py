#!/usr/bin/env python3
"""Output digests of the two operations lane mf-shaprfe moved onto native members (2026-11-23). A testing tool,
never product execution.

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
the CPU column, base origin/lane/member-fits 4d76a9e16e and the lane's head, and diff the files: equal lines mean
equal bits.

    PYTHONPATH=<repo>/python python3 tools/mf_shaprfe_digests.py > out.txt

Expected: every line equal, except `rfe_*_getter_coef` of QDA and GaussianNB, whose refusal words change from the
old path's (it fitted every round's clone, then failed reading a missing coef_) to scikit-learn's attrgetter words
("AttributeError: '<class>' object has no attribute 'coef_'").

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
    X = rng.normal(size=(n, d)).astype(np.float32)
    k = int(kind[3:])
    y = (np.digitize(X[:, 0] + 0.5 * X[:, 1], np.linspace(-1, 1, k - 1))).astype(np.int64)
    if positive:
        X = np.abs(X)
    return X, y


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


def _cases():
    cases = {}
    for key, (cls, mkw, positive) in _SHAP_MODELS.items():  # glue: the fixed configuration table
        cases["shap_k_" + key + "_identity"] = ("K", cls, mkw, positive, "clf2", {}, {}, 8)
        cases["shap_k_" + key + "_logit"] = ("K", cls, mkw, positive, "clf2", dict(link="logit"), {}, 8)
        cases["shap_p_" + key] = ("P", cls, mkw, positive, "clf2", dict(random_state=0), dict(npermutations=20), 8)
    cases["shap_k_logistic_clf3"] = ("K", "LogisticRegression", {}, False, "clf3", {}, {}, 8)
    cases["shap_p_logistic_many_rows"] = ("P", "LogisticRegression", {}, False, "clf2", dict(random_state=1),
                                          dict(npermutations=10), 97)
    cases["shap_k_dart_many_rows"] = ("K", "DARTClassifier", dict(n_estimators=12, random_state=0), False, "clf2",
                                      {}, {}, 97)
    for key, (cls, mkw, positive) in _RFE_MODELS.items():  # glue: the fixed configuration table
        for step in (1, 2):  # glue: the fixed configuration table
            for nsel in (None, 2):  # glue: the fixed configuration table
                name = "rfe_%s_step%d_n%s" % (key, step, "none" if nsel is None else nsel)
                cases[name] = ("R", cls, mkw, positive, "clf3", dict(step=step, n_features_to_select=nsel), {}, 0)
        cases["rfe_" + key + "_getter_coef"] = ("R", cls, mkw, positive, "clf3",
                                                dict(step=1, n_features_to_select=2, importance_getter="coef_"),
                                                {}, 0)
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


def _run(name):
    import numpy as np
    import mojolearn as ml
    which, cls, mkw, positive, kind, ekw, vkw, rows = _cases()[name]
    X, y = _data(kind, positive=positive)
    try:
        if which == "R":
            r = ml.RFE(getattr(ml, cls)(**mkw), **ekw).fit(X, y)
            proba = r.predict_proba(X[:10])
            return f"{name} sha256={_digest(r.ranking_, np.asarray(r.support_), proba)}"
        model = getattr(ml, cls)(**mkw).fit(X, y)
        explainer = (ml.KernelExplainer if which == "K" else ml.PermutationExplainer)(model, X[:20], **ekw)
        phi = np.asarray(explainer.shap_values(X[200:200 + rows].copy(), **vkw))
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
    workers = max(1, min(int(os.environ.get("MF_SHAPRFE_WORKERS", cores)), len(names)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for line in pool.map(_run, names):
            print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
