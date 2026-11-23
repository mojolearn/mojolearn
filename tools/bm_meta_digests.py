"""Before/after digests of the meta-estimator operations lanes bm-meta and bm-meta-2 split (VotingClassifier /
VotingRegressor: fit, predict, predict_proba, transform; bm-meta-2: MultiOutput*, OneVsRest, Bagging* and the OOB
score, Stacking*, CalibratedClassifierCV, AdaBoost*, micro one-vs-rest roc_auc; bm-meta-3: the Stacking fold that lacks a class), host column. Prints one line per case: name sha256 of the
output bytes (dtype and shape included). Test tool, never product runtime.

    python tools/bm_meta_digests.py > digests.txt
"""
import hashlib
import sys

import numpy as np

import mojolearn


def _digest(value):
    a = np.asarray(value)
    h = hashlib.sha256()
    h.update(f"{a.dtype.str}{a.shape}".encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:24]


def _data(seed, n=301, d=7, k=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    w = rng.normal(size=d)
    y = X @ w + 0.3 * rng.normal(size=n)
    edges = np.quantile(y, np.linspace(0, 1, k + 1)[1:-1])
    c = np.searchsorted(edges, y)
    return X, y.astype(np.float32), c, np.array(["a", "b", "c"])[c]


def _classifiers():
    return [
        ("lr", mojolearn.LogisticRegression(max_iter=200)),
        ("nb", mojolearn.GaussianNB()),
        ("rf", mojolearn.RandomForestClassifier(n_estimators=12, max_depth=5, random_state=3)),
        ("dt", mojolearn.DecisionTreeClassifier(max_depth=4, random_state=1)),
        ("kn", mojolearn.KNeighborsClassifier(n_neighbors=5)),
    ]


def _regressors():
    return [
        ("ridge", mojolearn.Ridge(alpha=0.5)),
        ("ols", mojolearn.LinearRegression()),
        ("rf", mojolearn.RandomForestRegressor(n_estimators=10, max_depth=5, random_state=4)),
        ("kn", mojolearn.KNeighborsRegressor(n_neighbors=7)),
    ]


def _safe(out, tag, thunk):
    """One case; a refusal is digested by its words (a refusal that changes is a difference)."""
    try:
        for name, value in thunk():
            out.append((tag + "_" + name, value))
    except Exception as e:  # noqa: BLE001 - the words are the digest
        out.append((tag + "_raised", f"{type(e).__name__}: {e}"))


def _clf_case(m, Xt, target, proba=True):
    rows = [("predict", _digest(m.predict(Xt))), ("score", repr(float(m.score(Xt[:60], target[:60]))))]
    if proba:
        rows.append(("proba", _digest(m.predict_proba(Xt))))
    return rows


def _more(out, X, y, c, labels, Xt):
    """bm-meta-2: the other meta-estimators (fit outputs and fitted state)."""
    Y = np.stack([y, -0.5 * y + 0.1, np.sin(y)], axis=1).astype(np.float32)
    C = np.stack([c, (c + 1) % 3, (c > 0).astype(np.int64)], axis=1)
    reg_members = [("ridge", lambda: mojolearn.Ridge(alpha=0.5)), ("ols", lambda: mojolearn.LinearRegression()),
                   ("kn", lambda: mojolearn.KNeighborsRegressor(n_neighbors=7)),
                   ("rf", lambda: mojolearn.RandomForestRegressor(n_estimators=6, max_depth=4, random_state=2))]
    clf_members = [("lr", lambda: mojolearn.LogisticRegression(max_iter=200)), ("nb", lambda: mojolearn.GaussianNB()),
                   ("kn", lambda: mojolearn.KNeighborsClassifier(n_neighbors=5)),
                   ("dt", lambda: mojolearn.DecisionTreeClassifier(max_depth=4, random_state=1)),
                   ("rf", lambda: mojolearn.RandomForestClassifier(n_estimators=6, max_depth=4, random_state=2))]
    for name, make in reg_members:
        def mo_reg(make=make):
            m = mojolearn.MultiOutputRegressor(make()).fit(X, Y)
            return [("predict", _digest(m.predict(Xt))), ("score", repr(float(m.score(Xt[:60], Y[:60]))))]
        _safe(out, f"mor_{name}", mo_reg)
    for name, make in clf_members:
        def mo_clf(make=make):
            m = mojolearn.MultiOutputClassifier(make()).fit(X, C)
            rows = [("predict", _digest(m.predict(Xt))), ("score", repr(float(m.score(Xt[:60], C[:60]))))]
            rows += [(f"proba{j}", _digest(p)) for j, p in enumerate(m.predict_proba(Xt))]
            return rows
        _safe(out, f"moc_{name}", mo_clf)
        for target_name, target in (("codes", c), ("labels", labels)):
            def ovr(make=make, target=target):
                m = mojolearn.OneVsRestClassifier(make()).fit(X, target)
                return _clf_case(m, Xt, target)
            _safe(out, f"ovr_{name}_{target_name}", ovr)
        def ovr_bin(make=make):
            b = (c > 0).astype(np.int64)
            m = mojolearn.OneVsRestClassifier(make()).fit(X, b)
            return _clf_case(m, Xt, b)
        _safe(out, f"ovrbin_{name}", ovr_bin)
        for feats, boot in ((1.0, True), (0.6, False)):
            def bag_c(make=make, feats=feats, boot=boot):
                m = mojolearn.BaggingClassifier(make(), n_estimators=7, max_samples=0.8, max_features=feats,
                                                bootstrap=boot, oob_score=boot, random_state=5).fit(X, labels)
                rows = _clf_case(m, Xt, labels)
                if boot:
                    rows.append(("oob", repr(float(m.oob_score_))))
                    rows.append(("oobdec", _digest(m.oob_decision_function_)))
                return rows
            _safe(out, f"bagc_{name}_{feats}_{boot}", bag_c)
        for method in ("sigmoid", "isotonic"):
            def cal(make=make, method=method):
                m = mojolearn.CalibratedClassifierCV(make(), method=method, cv=3).fit(X, labels)
                return _clf_case(m, Xt, labels)
            _safe(out, f"cal_{name}_{method}", cal)
    for name, make in reg_members:
        for feats, boot in ((1.0, True), (0.6, False)):
            def bag_r(make=make, feats=feats, boot=boot):
                m = mojolearn.BaggingRegressor(make(), n_estimators=7, max_samples=0.8, max_features=feats,
                                               bootstrap=boot, oob_score=boot, random_state=5).fit(X, y)
                rows = [("predict", _digest(m.predict(Xt))), ("score", repr(float(m.score(Xt[:60], y[:60]))))]
                if boot:
                    rows.append(("oob", repr(float(m.oob_score_))))
                    rows.append(("oobpred", _digest(m.oob_prediction_)))
                return rows
            _safe(out, f"bagr_{name}_{feats}_{boot}", bag_r)
    for passthrough in (False, True):
        def stack_c(passthrough=passthrough):
            m = mojolearn.StackingClassifier([(n, f()) for n, f in clf_members[:3]],
                                             final_estimator=mojolearn.LogisticRegression(max_iter=200), cv=3,
                                             passthrough=passthrough).fit(X, labels)
            return _clf_case(m, Xt, labels) + [("transform", _digest(m.transform(Xt)))]
        _safe(out, f"stackc_{passthrough}", stack_c)
        def stack_r(passthrough=passthrough):
            m = mojolearn.StackingRegressor([(n, f()) for n, f in reg_members[:3]],
                                            final_estimator=mojolearn.Ridge(alpha=1.0), cv=3,
                                            passthrough=passthrough).fit(X, y)
            return [("predict", _digest(m.predict(Xt))), ("transform", _digest(m.transform(Xt))),
                    ("score", repr(float(m.score(Xt[:60], y[:60]))))]
        _safe(out, f"stackr_{passthrough}", stack_r)
    # bm-meta-3: a cv fold whose training rows lack a class (class 3 only in the first rows, unshuffled KFold):
    # before, refused ("a member's output width changed between folds"); now placed at the classes (scikit-learn
    # `_enforce_prediction_order`). The one EXPECTED difference of the base/head comparison.
    for method in ("predict_proba", "decision_function"):
        def stack_gap(method=method):
            from mojolearn.model_selection import KFold
            lab4 = np.array(labels, dtype=object)
            lab4[:20] = "d"
            m = mojolearn.StackingClassifier([("lr", mojolearn.LogisticRegression(max_iter=200))],
                                             final_estimator=mojolearn.LogisticRegression(max_iter=200),
                                             cv=KFold(3), stack_method=method).fit(X, lab4)
            return [("predict", _digest(m.predict(Xt))), ("transform", _digest(m.transform(Xt)))]
        _safe(out, f"stackgap_{method}", stack_gap)
    for name, make in (("dt", lambda: mojolearn.DecisionTreeClassifier(max_depth=2, random_state=0)),
                       ("lr", lambda: mojolearn.LogisticRegression(max_iter=100))):
        def ada_c(make=make):
            m = mojolearn.AdaBoostClassifier(make(), n_estimators=12, learning_rate=0.7, random_state=3).fit(X, labels)
            return _clf_case(m, Xt, labels) + [("weights", _digest(m.estimator_weights_)),
                                                ("errors", _digest(m.estimator_errors_))]
        _safe(out, f"adac_{name}", ada_c)
    for loss in ("linear", "square", "exponential"):
        def ada_r(loss=loss):
            m = mojolearn.AdaBoostRegressor(mojolearn.DecisionTreeRegressor(max_depth=3, random_state=0),
                                            n_estimators=10, loss=loss, random_state=4).fit(X, y)
            return [("predict", _digest(m.predict(Xt))), ("score", repr(float(m.score(Xt[:60], y[:60])))),
                    ("weights", _digest(m.estimator_weights_)), ("errors", _digest(m.estimator_errors_))]
        _safe(out, f"adar_{loss}", ada_r)
    rng = np.random.default_rng(21)
    scores = rng.random((200, 4))
    scores /= scores.sum(axis=1, keepdims=True)
    yk = rng.integers(0, 4, size=200)
    sw = rng.random(200)
    for weighted in (False, True):
        def auc(weighted=weighted):
            kw = {"sample_weight": sw} if weighted else {}
            return [(avg, repr(float(mojolearn.metrics.roc_auc_score(yk, scores, multi_class="ovr", average=avg,
                                                                      **kw))))
                    for avg in ("micro", "macro", "weighted")]
        _safe(out, f"auc_{'w' if weighted else 'u'}", auc)


def main():
    out = []
    X, y, c, labels = _data(11)
    Xt, _, _, _ = _data(12, n=97)
    for target_name, target in (("codes", c), ("labels", labels)):
        for voting in ("soft", "hard"):
            for weights in (None, [1.0, 2.0, 0.5, 1.5, 0.25]):
                tag = f"vc_{target_name}_{voting}_{'w' if weights else 'u'}"
                m = mojolearn.VotingClassifier(_classifiers(), voting=voting, weights=weights).fit(X, target)
                out.append((tag + "_predict", _digest(m.predict(Xt))))
                out.append((tag + "_transform", _digest(m.transform(Xt))))
                if voting == "soft":
                    out.append((tag + "_proba", _digest(m.predict_proba(Xt))))
                out.append((tag + "_score", repr(float(m.score(Xt[:60], target[:60])))))
    for weights in (None, [2.0, 1.0, 0.5, 3.0]):
        tag = f"vr_{'w' if weights else 'u'}"
        m = mojolearn.VotingRegressor(_regressors(), weights=weights).fit(X, y)
        out.append((tag + "_predict", _digest(m.predict(Xt))))
        out.append((tag + "_transform", _digest(m.transform(Xt))))
        out.append((tag + "_score", repr(float(m.score(Xt[:60], y[:60])))))
    # Refusals keep their words.
    bad = Xt.copy()
    bad[3, 2] = np.nan
    for voting in ("soft", "hard"):
        m = mojolearn.VotingClassifier(_classifiers()[:2], voting=voting).fit(X, c)
        try:
            m.predict(bad)
            out.append((f"nan_{voting}", "no-refusal"))
        except Exception as e:  # noqa: BLE001 - the words are the digest
            out.append((f"nan_{voting}", f"{type(e).__name__}: {e}"))
    _more(out, X, y, c, labels, Xt)
    for name, value in out:
        print(name, value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
