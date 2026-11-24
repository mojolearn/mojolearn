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
ExtraTree-style splitter), DecisionTreeClassifier(random_state=0) on the integer fixture, and `tree_allbins`: ours with
n_bins=1024 against scikit-learn's plain DecisionTreeRegressor(random_state=0).

THE CAUSE OF THE `differs` ROWS (lane w3-fix-d, 2026-11-23; gate b7ee oracle.txt: 24 of 48 differ, every one a
continuous fixture, every integer fixture equal). Not the imputer and not split ties:
  * Seeding already matches scikit-learn: each step fits a NEW member state from the estimator's own words
    (x_prep/iterimp_run.mojo `_member_step`: `member_new(member.plan)`), the clone's random_state is never redrawn,
    and the imputer's stream reaches only the posterior draws and the neighbour draws (`_member_put`, `plan.draws`).
  * Our DecisionTreeRegressor searches splits over per-feature quantile bins (cuML's rule, n_bins=None -> 128; its
    public docstring in python/mojolearn/_expansion_trees.py: "thresholds can differ from scikit-learn's";
    ensemble/decisiontree/batched_levelalgo/quantiles.mojo: sample_count = min(rows, 4 * n_bins), bin edge
    idx = round((bin + 1) * sample_count / n_bins) - 1, deduplicated). A step's training block has about 340 rows
    of continuous values (400 rows, 15 percent missing), so about 340 distinct values per feature against 128 edges:
    most of scikit-learn's candidate partitions do not exist for ours, the chosen splits differ, and every imputed
    cell moves (cells = about every missing cell). The integer fixtures have 3 distinct values per feature, so every
    candidate partition is an edge on both and the rows are equal: that is why only the continuous rows differ.
  * splitter='random' draws its thresholds from our ExtraTrees stream (extratrees/, PCG), not scikit-learn's rand_r
    over an MT19937-derived seed: on continuous features it differs by construction; on 3-valued features every
    threshold in a gap gives the same partition (equal).
  * The imputer's storage dtype: given float64 X scikit-learn keeps Xt and every imputed target in float64; ours
    reads X as float32 (the estimator docstrings), as scikit-learn does when given float32 X (FLOAT_DTYPES keeps
    float32). The `_sk32` rows give scikit-learn the same float32 X, the like-for-like comparison.
THREE MORE CAUSES, each enough alone to keep a continuous row `differs` (lane w3-fix-d2, 2026-11-24, from the code):
  * Leaf values. Our regression histogram is Int32 fixed point (ensemble/decisiontree/batched_levelalgo/bins.mojo
    DEVIATION 101b: each label truncated to a power-of-two scale from checks/fixed_point.mojo `choose_scale`, about
    2^18 for a step's target), and the leaf is the dequantized sum over the count (objectives.mojo
    RegressionObjectiveFunction.SetLeafVector). scikit-learn's leaf is the float64 mean. A one-row leaf (the common
    leaf of a full-depth tree) is then trunc(y * 2^18) / 2^18 against y: about 4e-6 apart. Integer targets are exact
    in the fixed point, which is the other reason every int row is equal.
  * Thresholds. Ours publishes the bin edge (a training value, `x <= q[b]`); scikit-learn the midpoint
    (sklearn/tree/_splitter.pyx node_split_best: `feature_values[p_prev] / 2.0 + feature_values[p] / 2.0`). Same
    training partition, different routing of the imputed rows, whose values fall between training values.
  * Tie order. A node whose rows every feature separates the same way (any two-row node) gives every feature the SAME
    improvement. scikit-learn keeps the first feature its Fisher-Yates visit draws (rand_r over a seed from NumPy's
    MT19937 `randint`, with the permutation carried in the splitter from node to node in depth-first order); ours keeps
    the highest feature index (split.mojo `update`). The imputed rows routed through such a node follow different
    features.
EXPECTED on the current product: every int and ctree row equal; every continuous row (tree_*, tree_allbins_*,
tree_random_*, f64 and _sk32) `differs` by the causes above, tree_allbins with the smallest max_abs (about 1e-5 where
only the leaf quantization and the thresholds remain). `tree_random_*` uses scikit-learn's draw semantics already
(extratrees/impl/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo: uniform in [min, max) of the
node's feature, max -> min, FEATURE_THRESHOLD) from our own stream: its continuous rows differ by stream, not by rule.
No product route yet gives scikit-learn's values: that needs the decision recorded in
~/mojolearn-evidence/briefs-20261122/w3-fix-d2-RESULT.md.

Per case one line: `<case> equal` (bit-equal fit_transform, n_iter_, n_features_with_missing_ and step order), or
`<case> differs cells=<k>/<n> max_abs=<x> n_iter=<ours>/<sklearn> missing=<ours>/<sklearn> order=<same|differs>`,
or `<case> error ...`.

    PYTHONPATH=<repo>/python python3 tools/mf_gate6_iterimp_oracle.py > oracle.txt

Parallel plan: the cases are independent fits, one per worker process (at most the allocated cores).
"""
import itertools
import os
import sys
from concurrent.futures import ProcessPoolExecutor

ESTIMATORS = ("tree", "tree_allbins", "tree_random", "ctree")
SK_DTYPES = ("f64", "f32")  # the dtype scikit-learn is given (ours always reads float32)
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
    if name == "tree_allbins":
        # every distinct value of a step's training block an edge (n_bins >= its rows; 1024 is the tree's cap)
        return m.DecisionTreeRegressor(random_state=0, n_bins=1024) if lib == "ours" else \
            m.DecisionTreeRegressor(random_state=0)
    if name == "tree_random":
        return m.DecisionTreeRegressor(splitter="random", random_state=0)
    return m.DecisionTreeClassifier(random_state=0)


def _cases():
    cases = []
    for sk, est, fixture, order, skip in itertools.product(SK_DTYPES, ESTIMATORS, FIXTURES, ORDERS, (False, True)):
        if est == "ctree" and fixture == "cont":
            continue  # a classifier on a continuous target is refused on both (not a value case)
        cases.append((est, fixture, order, skip, sk))
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
    est, fixture, order, skip, sk = case
    # the f64 rows keep the gate's names (oracle.txt of gate b7ee); the f32 rows end in `_sk32`
    name = f"{est}_{fixture}_{order}_skip{int(skip)}" + ("_sk32" if sk == "f32" else "")
    X = _fixture(fixture, est == "ctree")
    try:
        ours = _fit("ours", est, X.copy(), order, skip)
    except Exception as e:  # noqa: BLE001 - the line records the refusal
        return f"{name} error ours {type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"
    try:
        theirs = _fit("sklearn", est, X.copy().astype(np.float64 if sk == "f64" else np.float32), order, skip)
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
