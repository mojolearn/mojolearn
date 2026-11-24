"""Before/after digests of lane mf-r9b (2026-11-23): the cv tools on the typed search RUN only, host column. One
line per configuration and output: name and sha256 of the output bytes (dtype and shape included); a refusal is
digested by its exception class and words. Base: origin/lane/member-fits-8 ea2e0c0139 (each tool kept the door's
units beside the RUN), head lane/mf-r9b (the RUN only). Test tool, never product runtime.

EXPECTED (base vs head):
- every `cv_*`, `cvs_*`, `cvp_*`, `perm_*` and `vc_ridge` / `lc_*` row the RUN served on base: equal digests
  (the same RUN; times are not digested).
- `vc_kmeans_noy`: base the units' scores, head the RUN's (equal values expected; base could differ only in
  float rounding of a score the units computed in Python's order).
- `vc_empty`: IndexError 'list index out of range' on both.
- `vc_multi`: base KeyError 'train_score'; head validate_params' InvalidParameterError words.
- `lc_empty`: base empty blocks; head NumPy's zero-size reduction words.
- `lc_frac_bad`: base the per-value absolute words; head the fractions words with [min, max] as %f.
- `lc_abs_bad`: base [a, a] of the first bad value; head [min, max].
- `lc_first_fold`: base refused ((0, 40]); head the first fold's n_max (50), the 40-row fold takes all its rows.
- `lc_dupes`: equal blocks; head also warns (not digested).

Configurations are independent: one process each across the host cores (the allocation's cores where the OS
reports them).

    python tools/mf_r9b_digests.py > digests.txt
"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import os
import sys
import warnings

import numpy as np


def _digest(a):
    a = np.ascontiguousarray(np.asarray(a))
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()[:24]


def _data():
    rng = np.random.default_rng(31)
    X = rng.normal(size=(60, 3)).astype(np.float32)
    yr = (X[:, 0] * 2.0 - X[:, 1] + 0.1 * rng.normal(size=60)).astype(np.float32)
    yc = (X[:, 0] + 0.3 * X[:, 1] > 0).astype(np.int64)
    return X, yr, yc


def _unit(name):
    out = []
    warnings.simplefilter("ignore")
    try:
        import mojolearn as ml
        from mojolearn import model_selection as ms
        X, yr, yc = _data()
        ridge = ml.Ridge(alpha=1.0, numeric_mode="identical")
        if name == "cv_ridge":
            got = ms.cross_validate(ridge, X, yr, cv=3, scoring=["r2", "neg_mean_squared_error"],
                                    return_train_score=True, return_indices=True)
            for key in ("test_r2", "train_r2", "test_neg_mean_squared_error"):  # glue: the digested columns
                out.append((f"{name}_{key}", _digest(got[key])))
            out.append((f"{name}_test_idx", _digest(np.concatenate(got["indices"]["test"]))))
        elif name == "cvs_knn":
            out.append((name, _digest(ms.cross_val_score(ml.KNeighborsClassifier(n_neighbors=3), X, yc, cv=4,
                                                         scoring="accuracy"))))
        elif name == "cvp_logreg_proba":
            out.append((name, _digest(ms.cross_val_predict(ml.LogisticRegression(max_iter=30), X, yc, cv=3,
                                                           method="predict_proba"))))
        elif name == "cvp_text_labels":
            labels = np.where(yc == 1, "yes", "no")
            pred = ms.cross_val_predict(ml.KNeighborsClassifier(n_neighbors=3), X, labels, cv=3)
            out.append((name, _digest(np.asarray(pred).astype("U3"))))
        elif name == "vc_ridge":
            tr, te = ms.validation_curve(ridge, X, yr, param_name="alpha", param_range=[0.1, 1.0, 10.0], cv=3)
            out += [(name + "_train", _digest(tr)), (name + "_test", _digest(te))]
        elif name == "vc_kmeans_noy":
            tr, te = ms.validation_curve(ml.KMeans(n_clusters=2, n_init=1, random_state=0), X, None,
                                         param_name="n_clusters", param_range=[2, 3], cv=3)
            out += [(name + "_train", _digest(tr)), (name + "_test", _digest(te))]
        elif name == "vc_empty":
            ms.validation_curve(ridge, X, yr, param_name="alpha", param_range=[], cv=3)
        elif name == "vc_multi":
            ms.validation_curve(ridge, X, yr, param_name="alpha", param_range=[1.0], cv=3, scoring=["r2"])
        elif name.startswith("lc_"):
            cv = 3
            sizes = {"lc_frac": [0.25, 0.5, 1.0], "lc_shuffle": [0.25, 1.0], "lc_dupes": [0.5, 0.5, 1.0],
                     "lc_empty": [], "lc_frac_bad": [0.0, 0.5], "lc_abs_bad": [5, 1000], "lc_first_fold": [45]}[name]
            if name == "lc_first_fold":
                rows = np.arange(60, dtype=np.int64)
                cv = [(rows[:50], rows[50:]), (rows[20:60], rows[:20])]
            got = ms.learning_curve(ridge, X, yr, cv=cv, train_sizes=sizes, shuffle=name == "lc_shuffle",
                                    random_state=0)
            out += [(name + "_sizes", _digest(got[0])), (name + "_train", _digest(got[1])),
                    (name + "_test", _digest(got[2]))]
        elif name == "perm_ridge":
            score, perms, p = ms.permutation_test_score(ridge, X, yr, cv=3, n_permutations=5, random_state=0)
            out.append((name, _digest(np.array([score, p] + list(perms), dtype=np.float64))))
        elif name == "perm_groups":
            groups = np.arange(60) % 4
            score, perms, p = ms.permutation_test_score(ml.KNeighborsClassifier(n_neighbors=3), X, yc,
                                                        groups=groups, cv=ms.GroupKFold(n_splits=4),
                                                        n_permutations=4, random_state=1)
            out.append((name, _digest(np.array([score, p] + list(perms), dtype=np.float64))))
        elif name == "perm_text":
            labels = list(np.where(yc == 1, "yes", "no"))
            score, perms, p = ms.permutation_test_score(ml.KNeighborsClassifier(n_neighbors=3), X, labels, cv=3,
                                                        n_permutations=3, random_state=2)
            out.append((name, _digest(np.array([score, p] + list(perms), dtype=np.float64))))
    except Exception as error:  # a refusal is digested by its class and words
        out.append((name + "_raised", hashlib.sha256(f"{type(error).__name__}: {error}".encode()).hexdigest()[:24]))
    return out


UNITS = ["cv_ridge", "cvs_knn", "cvp_logreg_proba", "cvp_text_labels", "vc_ridge", "vc_kmeans_noy", "vc_empty",
         "vc_multi", "lc_frac", "lc_shuffle", "lc_dupes", "lc_empty", "lc_frac_bad", "lc_abs_bad", "lc_first_fold",
         "perm_ridge", "perm_groups", "perm_text"]


def _cores():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def main():
    with ProcessPoolExecutor(max_workers=min(len(UNITS), _cores())) as pool:
        for lines in pool.map(_unit, UNITS):
            for name, value in lines:
                print(name, value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
