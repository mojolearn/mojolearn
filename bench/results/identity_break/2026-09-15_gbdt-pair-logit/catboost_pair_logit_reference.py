"""CatBoost CPU reference values for the gbdt-pair-logit lane's generated-pairs fit.

A correctness reference for the loss and metric VALUES, not a bitwise one: the
CatBoost reference trains on its CPU here (its GPU learner cannot run on this
Mac), computes in float64 and estimates leaves with its own CPU arithmetic, so
last digits are expected to differ. Nothing on our side is tuned to match.

Every option equals our fit's except the device: 20 trees, depth 6, learning
rate 0.03, l2 3, 128 GreedyLogSum borders over all rows, seed 0, no bootstrap,
random_strength 0, boost_from_average off, one Newton leaf iteration, Cosine,
SymmetricTree, nan_mode Min, one thread.

Run in the pixi `bench` environment (CatBoost 1.2.10), after the lane exists:

    PYTHONPATH=<worktree>/tools .pixi/envs/bench/bin/python catboost_pair_logit_reference.py out.json

The fixtures, the group sizes and the relevance grades come from
tools/identity_break.py itself (`fixture`, `_rank_groups`, `_relevance`), so
both sides read the same bytes.
"""
import json
import sys

import numpy as np
import catboost

import identity_break as ib

FIXTURES = ("base", "ties", "odd")
PARAMS = dict(
    loss_function="PairLogit",
    iterations=20,
    depth=6,
    learning_rate=0.03,
    l2_leaf_reg=3.0,
    border_count=128,
    feature_border_type="GreedyLogSum",
    random_seed=0,
    bootstrap_type="No",
    random_strength=0.0,
    boost_from_average=False,
    leaf_estimation_method="Newton",
    leaf_estimation_iterations=10,
    score_function="Cosine",
    grow_policy="SymmetricTree",
    boosting_type="Plain",
    nan_mode="Min",
    task_type="CPU",
    thread_count=1,
    use_best_model=False,
    allow_writing_files=False,
    verbose=False,
)
METRICS = ["PairLogit", "NDCG:type=Base", "DCG:type=Base"]


def main():
    out_path = sys.argv[1]
    record = dict(catboost=catboost.__version__, params=PARAMS, metrics=METRICS, fixtures={})
    for kind in FIXTURES:
        X, _yc, yr = ib.fixture(kind)
        groups = ib._rank_groups(X.shape[0])
        rel = ib._relevance(yr)
        pool = catboost.Pool(X, rel, group_id=groups)
        model = catboost.CatBoost(PARAMS)
        model.fit(pool)
        evals = model.eval_metrics(pool, METRICS)
        raw = model.predict(pool, prediction_type="RawFormulaVal")
        record["fixtures"][kind] = dict(
            n_rows=int(X.shape[0]),
            n_groups=int(len(np.unique(groups))),
            per_iteration={m: [float(v) for v in vals] for m, vals in evals.items()},
            final_raw_first8=[float(v) for v in raw[:8]],
        )
    with open(out_path, "w") as f:
        json.dump(record, f, indent=1, sort_keys=True)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
