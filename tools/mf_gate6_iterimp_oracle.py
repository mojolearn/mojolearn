#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Our IterativeImputer against scikit-learn's with a tree estimator (lane mf-gate6, 2026-11-23). A testing tool,
never product execution.

The gate's open question (nn-integrate-2 "The 3 open digest questions" 1): tools/mf_iterimp_digests.py tree_post0 (24)
and ctree_int (4) main lines differ between main d67c21591e and the member route while ridge, linreg and knn are equal.
Reading the code (scikit-learn 1.9.1 sklearn/impute/_iterative.py): each step fits `clone(self._estimator)` (:413), so
every step's tree keeps the estimator's own random_state (0 here) and never draws from the imputer's `random_state_`,
which only `truncnorm.rvs` (:460, sample_posterior), `choice` (:504, n_nearest_features) and `shuffle` (:550, order
'random') consume. skip_complete defaults to False (:326); with True only features with missing entries get a step
(:534-537). Ours: the member plan carries the tree's own words (its random_state among them,
providers/python/forest_member_common_boundary_door.mojo `forest_member_admit_common`); x_prep/iterimp_run.mojo never
hands its splitmix64 stream to a member fit. So the extra steps of skip_complete=False cannot move a tree's draws.

This script settles it on values. The deterministic configurations only (imputation_order ascending / roman / arabic,
n_nearest_features None, sample_posterior False: the imputer's own stream is ours, not NumPy's MT19937, so a 'random'
order or neighbour draw is never compared), skip_complete True / False, on the digest's fixtures
(mf_iterimp_digests.py `_data`, `_data_int`) plus one with a complete column (where skip_complete matters), for
DecisionTreeRegressor(random_state=0) (member route), DecisionTreeRegressor(splitter='random', random_state=0) (the
ExtraTree-style splitter: the door rounds) and DecisionTreeClassifier(random_state=0) on the integer fixture.

Per case one line: `<case> equal` (bit-equal fit_transform, n_iter_, n_features_with_missing_ and step order), or
`<case> differs cells=<k>/<n> max_abs=<x> n_iter=<ours>/<sklearn> order=<same|differs>`, or `<case> error ...`. A
tree split that ties between features is broken by each library's own feature permutation (scikit-learn's MT19937,
ours): a `differs` row on a fixture with tied splits is a tie, not an order fault; the skip_complete pairs must then
differ in the same way (same `order`, same n_features_with_missing_ as scikit-learn).

    PYTHONPATH=<repo>/python python3 tools/mf_gate6_iterimp_oracle.py > oracle.txt

Parallel plan: the cases are independent fits, one per worker process (at most the allocated cores).
"""
import itertools
import os
import sys
from concurrent.futures import ProcessPoolExecutor

ESTIMATORS = ("tree", "tree_random", "ctree")
ORDERS = ("ascending", "roman", "arabic")
FIXTURES = ("cont", "int", "complete")


def _data(seed=0, n=400, d=5):
    # tools/mf_iterimp_digests.py `_data`
    import numpy as np
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, 2))
    X = (Z @ rng.standard_normal((2, d)) + 0.1 * rng.standard_normal((n, d))).astype(np.float32)
    X[rng.random((n, d)) < 0.15] = np.nan
    return X


def _data_int(seed=0, n=400, d=4):
    # tools/mf_iterimp_digests.py `_data_int`
    import numpy as np
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 3, n)
    X = np.stack([labels + (j % 2) for j in range(d)], axis=1).astype(np.float32)
    rows = rng.random(n) < 0.6
    cols = rng.integers(0, d, n)
    X[rows, cols[rows]] = np.nan
    return X


def _fixture(name, classifier):
    import numpy as np
    if name == "int" or classifier:
        X = _data_int(0)
    else:
        X = _data(0)
    if name == "complete":
        # column 1 complete: skip_complete decides whether it gets a step
        full = np.nan_to_num(_data_int(5) if classifier else _data(5))
        X[:, 1] = full[:, 1]
    return np.ascontiguousarray(X)


def _estimator(lib, name):
    if lib == "ours":
        import mojolearn as m
    else:
        import sklearn.tree as m
    if name == "tree":
        return m.DecisionTreeRegressor(random_state=0)
    if name == "tree_random":
        return m.DecisionTreeRegressor(splitter="random", random_state=0)
    return m.DecisionTreeClassifier(random_state=0)


def _cases():
    cases = []
    for est, fixture, order, skip in itertools.product(ESTIMATORS, FIXTURES, ORDERS, (False, True)):
        if est == "ctree" and fixture == "cont":
            continue  # a classifier on a continuous target is refused on both (not a value case)
        cases.append((est, fixture, order, skip))
    return cases


def _fit(lib, est, X, order, skip):
    import numpy as np
    if lib == "ours":
        from mojolearn.impute import IterativeImputer
    else:
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer
    m = IterativeImputer(estimator=_estimator(lib, est), imputation_order=order, skip_complete=skip, max_iter=4,
                         random_state=3)
    Xt = np.asarray(m.fit_transform(X), dtype=np.float32)
    steps = [int(step[0]) for step in m.imputation_sequence_]
    return Xt, int(m.n_iter_), int(m.n_features_with_missing_), steps


def _run(case):
    import numpy as np
    est, fixture, order, skip = case
    name = f"{est}_{fixture}_{order}_skip{int(skip)}"
    X = _fixture(fixture, est == "ctree")
    try:
        ours = _fit("ours", est, X.copy(), order, skip)
    except Exception as e:  # noqa: BLE001 - the line records the refusal
        return f"{name} error ours {type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"
    try:
        theirs = _fit("sklearn", est, X.copy().astype(np.float64), order, skip)
    except Exception as e:  # noqa: BLE001 - the line records the refusal
        return f"{name} error sklearn {type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"
    a, b = ours[0], theirs[0].astype(np.float32)
    same_order = ours[3] == theirs[3] and ours[2] == theirs[2]
    if a.shape == b.shape and np.array_equal(a, b) and ours[1] == theirs[1] and same_order:
        return f"{name} equal"
    if a.shape != b.shape:
        return f"{name} differs shape={a.shape}/{b.shape}"
    cells = int(np.sum(a != b))
    worst = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if cells else 0.0
    return (f"{name} differs cells={cells}/{a.size} max_abs={worst:.6g} n_iter={ours[1]}/{theirs[1]} "
            f"missing={ours[2]}/{theirs[2]} order={'same' if same_order else 'differs'}")


def main():
    cases = _cases()
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:
        cores = os.cpu_count() or 1
    workers = max(1, min(cores, len(cases), int(os.environ.get("MF_GATE6_WORKERS", cores))))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for line in pool.map(_run, cases):
            print(line, flush=True)
    print(f"# cases={len(cases)} workers={workers} of {cores} cores", file=sys.stderr)


if __name__ == "__main__":
    main()
