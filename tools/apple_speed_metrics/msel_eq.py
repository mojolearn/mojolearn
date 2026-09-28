# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane metrics-apple3: one digest per model selection result, for the
routes the lane changed (the fold-row cache of the searches, the curves and
the permutation test, the scorers' shared predictions, learning_curve's row
prefixes, cross_val_predict's native assembly). The job runs it in the base
tree, in the head tree under MOJOLEARN_MSEL3_BEFORE=1 (every definition)
and in the head tree as shipped, per numeric mode, and requires every line
equal. Lines: `MEQ <case> <digest>`.

    python tools/apple_speed_metrics/msel_eq.py --tree <checkout> [--big 0|1]
"""
import hashlib
import os
import sys

import numpy as np

_tree = sys.argv[sys.argv.index("--tree") + 1] if "--tree" in sys.argv else \
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
_big = "--big" not in sys.argv or sys.argv[sys.argv.index("--big") + 1] != "0"
sys.path.insert(0, os.path.join(_tree, "python"))


def dig(v):
    h = hashlib.sha256()

    def walk(x):
        if isinstance(x, dict):
            for k in sorted(x):
                if "time" in str(k):
                    continue
                h.update(str(k).encode())
                walk(x[k])
            return
        if isinstance(x, (list, tuple)):
            h.update(b"[%d" % len(x))
            for e in x:
                walk(e)
            return
        if isinstance(x, float):
            h.update(np.float64(x).tobytes())
            return
        if isinstance(x, (int, str, type(None))):
            h.update(repr(x).encode())
            return
        a = np.asarray(x.tolist() if hasattr(x, "tolist") else x)
        h.update(str(a.dtype).encode() + str(a.shape).encode())
        h.update(np.ascontiguousarray(a).tobytes() if a.dtype != object else repr(a.tolist()).encode())
    walk(v)
    return h.hexdigest()[:16]


class HashEstimator:
    """Its score and its predictions are digests of every byte it was fit
    on and asked about: any change in a fold's rows, their order, or the
    words of X or y moves them."""
    _estimator_type = "classifier"

    def __init__(self, salt=0):
        self.salt = salt

    def get_params(self, deep=False):
        return {"salt": self.salt}

    def set_params(self, **p):
        self.salt = p.get("salt", self.salt)
        return self

    def fit(self, X, y):
        a, x = np.asarray(y), np.asarray(X)
        self.seen_ = hashlib.blake2b(
            str((a.dtype, a.shape, x.dtype, x.shape, self.salt)).encode() + a.tobytes() + x.tobytes(),
            digest_size=8).digest()
        self.classes_ = [0, 1]
        return self

    def _u(self, X, tag):
        x = np.asarray(X)
        d = hashlib.blake2b(self.seen_ + tag + str(x.shape).encode() + x.tobytes(), digest_size=8).digest()
        r = np.random.RandomState(int.from_bytes(d[:4], "little"))
        return r.rand(x.shape[0])

    def predict(self, X):
        return (self._u(X, b"p") < 0.5).astype(np.int64)

    def predict_proba(self, X):
        u = self._u(X, b"q").astype(np.float32)
        return np.stack([np.float32(1) - u, u], axis=1)

    def score(self, X, y):
        a = np.asarray(y)
        d = hashlib.blake2b(self.seen_ + a.tobytes() + np.asarray(X).tobytes(), digest_size=6).digest()
        return int.from_bytes(d, "little") / float(1 << 48)


def main():
    import warnings
    warnings.simplefilter("ignore")
    import mojolearn as ml
    import mojolearn.model_selection as S
    rs = np.random.RandomState(11)
    cases = []

    def callable_scorer(est, X, y):
        return float(np.mean(np.asarray(est.predict(X)) == np.asarray(y)))

    sizes = (300, 5000, 70001) if _big else (300, 5000)
    for n in sizes:
        X = rs.randn(n, 6).astype(np.float32)
        yr = (X @ rs.randn(6) + 0.1 * rs.randn(n)).astype(np.float32)
        yb = (X[:, 0] + 0.5 * rs.randn(n) > 0).astype(np.int64)
        kf = lambda seed=1: S.KFold(4, shuffle=True, random_state=seed)
        t = "_%d" % n
        H, R, G = HashEstimator, ml.Ridge, ml.GaussianNB
        cases += [
            ("cv_multi_hash" + t, lambda X=X, y=yb: S.cross_validate(
                H(), X, y, cv=kf(), scoring=["roc_auc", "accuracy", "neg_log_loss", "average_precision", "f1"],
                return_train_score=True)),
            ("cv_multi_gnb" + t, lambda X=X, y=yb: S.cross_validate(
                G(), X, y, cv=5, scoring=["roc_auc", "accuracy", "neg_log_loss", "neg_brier_score"],
                return_train_score=True)),
            ("cv_multi_ridge" + t, lambda X=X, y=yr: S.cross_validate(
                R(), X, y, cv=kf(2), scoring=["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"])),
            ("cv_dict_callable" + t, lambda X=X, y=yb: S.cross_validate(
                H(), X, y, cv=kf(3), scoring={"a": "accuracy", "b": callable_scorer, "c": "roc_auc"},
                return_train_score=True)),
            ("cv_single_hash" + t, lambda X=X, y=yb: S.cross_validate(H(), X, y, cv=kf(4), return_train_score=True)),
            ("grid_hash" + t, lambda X=X, y=yb: S.GridSearchCV(
                H(), {"salt": [0, 1, 2]}, cv=kf(5), scoring=["accuracy", "roc_auc"], refit=False,
                return_train_score=True).fit(X, y).cv_results_),
            ("grid_hash_default_cv" + t, lambda X=X, y=yb: S.GridSearchCV(
                H(), {"salt": [3, 4]}).fit(X, y).cv_results_),
            ("grid_ridge" + t, lambda X=X, y=yr: S.GridSearchCV(
                R(), {"alpha": [0.01, 1.0, 100.0]}, cv=kf(6)).fit(X, y).cv_results_),
            ("rand_hash" + t, lambda X=X, y=yb: S.RandomizedSearchCV(
                H(), {"salt": list(range(20))}, n_iter=4, cv=kf(7), random_state=3).fit(X, y).cv_results_),
            ("valcurve_hash" + t, lambda X=X, y=yb: S.validation_curve(
                H(), X, y, param_name="salt", param_range=[0, 5, 9], cv=kf(8))),
            ("valcurve_ridge" + t, lambda X=X, y=yr: S.validation_curve(
                R(), X, y, param_name="alpha", param_range=[0.1, 10.0], cv=3, scoring="neg_mean_squared_error")),
            ("learn_hash" + t, lambda X=X, y=yb: S.learning_curve(H(), X, y, cv=kf(9))),
            ("learn_hash_shuffle" + t, lambda X=X, y=yb: S.learning_curve(
                H(), X, y, cv=kf(10), shuffle=True, random_state=5, train_sizes=(0.2, 0.5, 1.0))),
            ("learn_hash_ints" + t, lambda X=X, y=yb, n=n: S.learning_curve(
                H(), X, y, cv=3, shuffle=True, random_state=6, train_sizes=(7, n // 3, n // 2), scoring="accuracy")),
            ("learn_ridge" + t, lambda X=X, y=yr: S.learning_curve(R(), X, y, cv=kf(11), scoring="r2")),
            ("learn_gnb_shuffle" + t, lambda X=X, y=yb: S.learning_curve(
                G(), X, y, cv=4, shuffle=True, random_state=2, scoring="roc_auc")),
            ("cvp_ridge" + t, lambda X=X, y=yr: S.cross_val_predict(R(), X, y, cv=kf(12))),
            ("cvp_ridge_default" + t, lambda X=X, y=yr: S.cross_val_predict(R(), X, y)),
            ("cvp_gnb" + t, lambda X=X, y=yb: S.cross_val_predict(G(), X, y, cv=kf(13))),
            ("cvp_gnb_proba" + t, lambda X=X, y=yb: S.cross_val_predict(G(), X, y, cv=5, method="predict_proba")),
            ("cvp_hash" + t, lambda X=X, y=yb: S.cross_val_predict(H(), X, y, cv=kf(14))),
            ("cvp_hash_proba" + t, lambda X=X, y=yb: S.cross_val_predict(H(), X, y, cv=kf(15), method="predict_proba")),
            ("cvp_not_partition" + t, lambda X=X, y=yb: S.cross_val_predict(
                H(), X, y, cv=S.ShuffleSplit(3, test_size=0.25, random_state=1))),
            ("perm_hash_kfold" + t, lambda X=X, y=yb: S.permutation_test_score(
                H(), X, y, cv=S.KFold(5), n_permutations=4, random_state=7)),
            ("perm_hash_stratified" + t, lambda X=X, y=yb: S.permutation_test_score(
                H(), X, y, cv=5, n_permutations=3, random_state=8)),
            ("perm_gnb_kfold" + t, lambda X=X, y=yb: S.permutation_test_score(
                G(), X, y, cv=kf(16), n_permutations=3, random_state=9, scoring="roc_auc")),
        ]
    for name, fn in cases:
        try:
            print("MEQ %s %s" % (name, dig(fn())), flush=True)
        except Exception as e:
            print("MEQ %s RAISED %s %s" % (name, type(e).__name__, str(e)[:80]), flush=True)


if __name__ == "__main__":
    main()
