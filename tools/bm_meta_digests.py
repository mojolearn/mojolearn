"""Before/after digests of the meta-estimator operations lane bm-meta splits (VotingClassifier / VotingRegressor:
predict, predict_proba, transform, fitted state), host column. Prints one line per case: name sha256 of the
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
    for name, value in out:
        print(name, value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
