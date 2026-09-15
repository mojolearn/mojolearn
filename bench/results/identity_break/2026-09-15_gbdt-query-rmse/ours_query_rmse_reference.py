"""Our values for the gbdt-query-rmse configuration, for the CatBoost CPU reference
comparison (compare_query_rmse_reference.py). Metal route, identical tier.

For each fixture: the learn loss curve (our `loss_curve_`, the per-row mean of
w * (residual - query mean)^2, which is the square of CatBoost's QueryRMSE metric
at unit weights), the final QueryRMSE computed from the predictions in float64 by
the reference's CPU metric definition (`libs/metrics/metric.cpp:2622-2690`), and
NDCG and DCG (type Base, LogPosition denominator, no top) computed in float64 by
the reference's CPU definitions (`libs/metrics/dcg.cpp`, ties broken by
`doc_comparator.h`: higher prediction first, equal predictions lower target
first), averaged over queries as the reference's additive metric does.
"""
import json
import sys

import numpy as np

sys.path.insert(0, sys.argv[2] if len(sys.argv) > 2 else
                "/private/tmp/claude-501/-Users-andrewhendel-CascadeProjects-mojolearn/e52730fc-0ee7-4bf3-8741-5fa0e0f1876f/scratchpad/wt-gbdt-rank/tools")
import identity_break as ib  # noqa: E402

import mojolearn as ml  # noqa: E402
from mojolearn._cpu_reference import reference_training  # noqa: E402

FIXTURES = ("base", "ties", "odd")


def query_bounds(groups):
    starts = [0] + [i for i in range(1, len(groups)) if groups[i] != groups[i - 1]] + [len(groups)]
    return list(zip(starts[:-1], starts[1:]))


def query_rmse(pred, target, bounds):
    num = 0.0
    den = 0.0
    for b, e in bounds:
        resid = target[b:e].astype(np.float64) - pred[b:e].astype(np.float64)
        avg = resid.sum() / (e - b)
        num += float(((resid - avg) ** 2).sum())
        den += float(e - b)
    return float(np.sqrt(num / (den + 1e-38)))


def dcg_values(pred, target, bounds):
    """Per-query (DCG, NDCG), Base type, LogPosition decay, all positions."""
    dcgs, ndcgs = [], []
    for b, e in bounds:
        p = pred[b:e].astype(np.float64)
        t = target[b:e].astype(np.float64)
        n = e - b
        decay = np.ones(n)
        decay[1:] = 1.0 / np.log2(np.arange(1, n) + 2.0)
        # CompareDocs: approx descending, then target ascending; a stable sort keeps index order last
        order = sorted(range(n), key=lambda i: (-p[i], t[i]))
        dcg = float(np.dot(t[order], decay))
        ideal = np.sort(t)[::-1]
        idcg = float(np.dot(ideal, decay))
        dcgs.append(dcg)
        ndcgs.append(dcg / idcg if idcg > 0 else 1.0)
    return float(np.mean(dcgs)), float(np.mean(ndcgs))


def main():
    out_path = sys.argv[1]
    record = dict(fixtures={})
    with reference_training():
        for kind in FIXTURES:
            X, _yc, yr = ib.fixture(kind)
            groups = ib._rank_groups(X.shape[0])
            rel = ib._relevance(yr)
            m = ml.GradientBoosting(n_estimators=20, max_depth=6, loss="QueryRMSE").fit(X, rel, group_id=groups)
            pred = np.asarray(m.predict(X), dtype=np.float32)
            bounds = query_bounds(groups)
            dcg, ndcg = dcg_values(pred, rel, bounds)
            record["fixtures"][kind] = dict(
                n_rows=int(X.shape[0]),
                n_groups=len(bounds),
                loss_curve=[float(v) for v in m.loss_curve_],
                final_query_rmse=query_rmse(pred, rel, bounds),
                final_dcg=dcg,
                final_ndcg=ndcg,
                final_raw_first8=[float(v) for v in pred[:8]],
            )
    with open(out_path, "w") as f:
        json.dump(record, f, indent=1, sort_keys=True)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
