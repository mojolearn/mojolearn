"""Before/after digests of the member entries lane mf-entries5 added (2026-11-23), host column. One line per
configuration and output: name and sha256 of the output bytes (dtype and shape included); a refusal is digested by
its words. Base: origin/lane/member-fits-5 0ed9b30aa2 (LabelPropagation / LabelSpreading carried no member mark:
every meta-estimator called their own doors), head lane/mf-entries5 (XNeighbors kinds 7 and 8 run as native
members wherever the labels their fit sees are class codes without -1). Test tool, never product runtime.

Configurations, for LabelPropagation (kernel='knn' and 'rbf') and LabelSpreading (kernel='knn' and 'rbf'):
  * `*_bag_*`: BaggingClassifier (bootstrap draws; a draw missing a class is refused by the member and fitted by
    the caller's own loop, as before) - predict_proba, and each member's label_distributions_ / transduction_.
  * `*_vote_*`, `*_stack_*`, `*_ovr_*`: soft VotingClassifier, StackingClassifier, OneVsRestClassifier.
  * `*_cal_*`: CalibratedClassifierCV(cv=3) over y without -1 (members) and over y with -1 (`*_cal_marked_*`: the
    door, unchanged by design).
  * `*_search_*`: GridSearchCV over y without -1 (members) and with -1 (`*_search_marked_*`: the Python units).

EXPECTED: every line equal. The member runs `lp_fit_run` / `lp_query_run`, the door's bodies, on the codes the
caller's door fit was handed, with the marker word = the class count (the door's word when no row is -1). Lines
that may differ, stated: `*_raised` lines whose words change from a door refusal to the member's refusal of the same
input. Configurations are independent: one process each across the host cores.

    python tools/mf_entries5_digests.py > digests.txt
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


def _data(seed, n, k, d=3):
    rng = np.random.default_rng(seed)
    X = rng.random((n, d), dtype=np.float32)
    edges = np.quantile(X[:, 0], np.linspace(0, 1, k + 1)[1:-1])
    return X, np.searchsorted(edges, X[:, 0]).astype(np.int64)


def _make(name, kernel):
    import mojolearn
    if name == "lp":
        return mojolearn.LabelPropagation(kernel=kernel, max_iter=200)
    return mojolearn.LabelSpreading(kernel=kernel, alpha=0.2)


def _unit(arg):
    name, kernel, meta = arg
    import mojolearn
    from mojolearn.calibration import CalibratedClassifierCV
    from mojolearn.model_selection import GridSearchCV
    X, y = _data(7, 90, 3)
    marked = y.copy()
    marked[::4] = -1
    tag = f"{name}_{kernel}_{meta}"
    out = []
    try:
        if meta == "bag":
            m = mojolearn.BaggingClassifier(_make(name, kernel), n_estimators=3, random_state=0).fit(X, y)
            out.append((tag + "_proba", _digest(m.predict_proba(X))))
            for i, est in enumerate(m.estimators_):
                out.append((f"{tag}{i}_dist", _digest(est.label_distributions_)))
                out.append((f"{tag}{i}_trans", _digest(est.transduction_)))
                out.append((f"{tag}{i}_n_iter", repr(int(est.n_iter_))))
        elif meta == "vote":
            m = mojolearn.VotingClassifier([("a", _make(name, kernel)), ("b", _make(name, kernel))],
                                           voting="soft").fit(X, y)
            out.append((tag + "_proba", _digest(m.predict_proba(X))))
        elif meta == "stack":
            m = mojolearn.StackingClassifier([("a", _make(name, kernel))], cv=3).fit(X, y)
            out.append((tag + "_predict", _digest(m.predict(X))))
        elif meta == "ovr":
            m = mojolearn.OneVsRestClassifier(_make(name, kernel)).fit(X, y)
            out.append((tag + "_proba", _digest(m.predict_proba(X))))
        elif meta in ("cal", "cal_marked"):
            target = marked if meta == "cal_marked" else y
            m = CalibratedClassifierCV(_make(name, kernel), cv=3).fit(X, target)
            out.append((tag + "_proba", _digest(m.predict_proba(X))))
        else:
            target = marked if meta == "search_marked" else y
            grid = {"max_iter": [5, 50]} if name == "lp" else {"alpha": [0.2, 0.5]}
            m = GridSearchCV(_make(name, kernel), grid, cv=3).fit(X, target)
            out.append((tag + "_scores", _digest(np.asarray(m.cv_results_["mean_test_score"], dtype=np.float64))))
            out.append((tag + "_best_dist", _digest(m.best_estimator_.label_distributions_)))
    except Exception as e:  # noqa: BLE001 - the words are the digest
        out.append((tag + "_raised", f"{type(e).__name__}: {e}"))
    return out


UNITS = [(n, k, m) for n in ("lp", "ls") for k in ("knn", "rbf")
         for m in ("bag", "vote", "stack", "ovr", "cal", "cal_marked", "search", "search_marked")]


def main():
    with ProcessPoolExecutor(max_workers=min(len(UNITS), os.cpu_count() or 1)) as pool:
        for lines in pool.map(_unit, UNITS):
            for name, value in lines:
                print(name, value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
