"""The CatBoost CPU reference and our values for the gbdt-yeti-rank lane's 20-tree fit.

A QUALITY reference only. CatBoost's CPU YetiRank samples its permutations with
its own generator (`private/libs/algo/yetirank_helpers.cpp`, TFastRng64 per
block), not the GPU kernel's per-task LCG streams this implementation follows,
and YetiRank has no loss value on either side (the GPU target writes 0; its
score metric is PFound). So only the final NDCG and DCG of the two fits are
compared, as exact differences; nothing is tuned and no threshold is applied.

Every option equals our fit's except the device: 20 trees, depth 6, learning
rate 0.03, l2 0 (the reference's YetiRank default, `catboost_options.cpp:
166-172`), 128 GreedyLogSum borders, seed 0, no bootstrap, random_strength 0,
boost_from_average off, Newton at one iteration, Cosine, SymmetricTree,
nan_mode Min, one thread, permutations 10, decay 0.85.

    PYTHONPATH=<worktree>/tools .pixi/envs/bench/bin/python yeti_rank_reference.py catboost out.json
    PYTHONPATH=<worktree>/python .pixi/envs/test/bin/python yeti_rank_reference.py ours out.json
    python yeti_rank_reference.py compare ours.json catboost.json
"""
import json
import sys

import numpy as np

FIXTURES = ("base", "ties", "odd")
PARAMS = dict(
    loss_function="YetiRank:permutations=10;decay=0.85",
    iterations=20,
    depth=6,
    learning_rate=0.03,
    l2_leaf_reg=0.0,
    border_count=128,
    feature_border_type="GreedyLogSum",
    random_seed=0,
    bootstrap_type="No",
    random_strength=0.0,
    boost_from_average=False,
    leaf_estimation_iterations=1,
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
METRICS = ["NDCG:type=Base", "DCG:type=Base"]


def _ib():
    import identity_break as ib
    return ib


def query_bounds(groups):
    starts = [0] + [i for i in range(1, len(groups)) if groups[i] != groups[i - 1]] + [len(groups)]
    return list(zip(starts[:-1], starts[1:]))


def dcg_values(pred, target, bounds):
    """Per-query mean (DCG, NDCG), Base type, LogPosition decay, all positions;
    ties: higher prediction first, equal predictions lower target first."""
    dcgs, ndcgs = [], []
    for b, e in bounds:
        p = pred[b:e].astype(np.float64)
        t = target[b:e].astype(np.float64)
        n = e - b
        decay = np.ones(n)
        decay[1:] = 1.0 / np.log2(np.arange(1, n) + 2.0)
        order = sorted(range(n), key=lambda i: (-p[i], t[i]))
        dcg = float(np.dot(t[order], decay))
        idcg = float(np.dot(np.sort(t)[::-1], decay))
        dcgs.append(dcg)
        ndcgs.append(dcg / idcg if idcg > 0 else 1.0)
    return float(np.mean(dcgs)), float(np.mean(ndcgs))


def run_catboost(out_path):
    import catboost
    ib = _ib()
    record = dict(catboost=catboost.__version__, params=PARAMS, metrics=METRICS, fixtures={})
    for kind in FIXTURES:
        X, _yc, yr = ib.fixture(kind)
        groups = ib._rank_groups(X.shape[0])
        rel = ib._relevance(yr)
        model = catboost.CatBoost(PARAMS)
        pool = catboost.Pool(X, rel, group_id=groups)
        model.fit(pool)
        evals = model.eval_metrics(pool, METRICS)
        record["fixtures"][kind] = dict(
            n_rows=int(X.shape[0]),
            final={m: float(v[-1]) for m, v in evals.items()},
        )
    json.dump(record, open(out_path, "w"), indent=1, sort_keys=True)


def run_ours(out_path):
    import mojolearn as ml
    from mojolearn._cpu_reference import reference_training
    ib = _ib()
    record = dict(fixtures={})
    with reference_training():
        for kind in FIXTURES:
            X, _yc, yr = ib.fixture(kind)
            groups = ib._rank_groups(X.shape[0])
            rel = ib._relevance(yr)
            m = ml.GradientBoosting(n_estimators=20, max_depth=6, loss="YetiRank").fit(X, rel, group_id=groups)
            pred = np.asarray(m.predict(X), dtype=np.float32)
            dcg, ndcg = dcg_values(pred, rel, query_bounds(groups))
            record["fixtures"][kind] = dict(n_rows=int(X.shape[0]), final_dcg=dcg, final_ndcg=ndcg)
    json.dump(record, open(out_path, "w"), indent=1, sort_keys=True)


def compare(ours_path, ref_path):
    ours = json.load(open(ours_path))
    ref = json.load(open(ref_path))
    print(f"CatBoost {ref.get('catboost')} CPU YetiRank vs our fit: final NDCG and DCG only")
    for kind, o in ours["fixtures"].items():
        r = ref["fixtures"][kind]["final"]
        n = next(v for k, v in r.items() if k.startswith("NDCG"))
        d = next(v for k, v in r.items() if k.startswith("DCG"))
        print(f"{kind}: NDCG ours {o['final_ndcg']:.9g} reference {n:.9g} diff {o['final_ndcg'] - n:+.3e};"
              f" DCG ours {o['final_dcg']:.9g} reference {d:.9g} diff {o['final_dcg'] - d:+.3e}")


if __name__ == "__main__":
    if sys.argv[1] == "catboost":
        run_catboost(sys.argv[2])
    elif sys.argv[1] == "ours":
        run_ours(sys.argv[2])
    else:
        compare(sys.argv[2], sys.argv[3])
