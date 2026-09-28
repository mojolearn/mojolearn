"""The trees lane's CPU speed bench (lane/trees-cpu, 2026-09-28).

Times every trees algorithm's host (CPU) path through the public classes,
once per cell, and prints a sha256 of every output (the fitted model's
predictions on held-out rows, and the model text where there is one), so one
run is both the before/after timing and the bit check: run it at
MOJOLEARN_CPU_THREADS = 1, 3 and unset and the digests must match; before and
after a change they must match too; and on the GPU (no MOJOLEARN_VENDOR) the
digests are the GPU column's, which the CPU column must equal.

    MOJOLEARN_VENDOR=cpu MOJOLEARN_HOST_DIR=python/mojolearn/host \
      pixi run python bench/speed/trees_cpu_speed.py --data higgs.npz --algos all --out r.json

Data: the first `--n` rows of `--data` (an .npz with `X` and `y`, staged from
R2 by tools/dataset_store.sh; HIGGS is 11M x 28 with a binary label),
float32; held-out rows are the next `--m`. The regression target is a fixed
function of the columns (host float64, then float32). Without `--data`, a
seeded Gaussian problem (a smoke only). `--rows name=N,...` caps one
algorithm's training rows (the O(n^2) or per-row Python wrappers).
"""
import argparse
import hashlib
import json
import os
import tempfile
import time

import numpy as np


def digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        if isinstance(a, str):
            h.update(a.encode())
            continue
        a = np.asarray(a.toarray() if hasattr(a, "toarray") else a)
        h.update(str(a.dtype).encode() + str(a.shape).encode() + np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def load(args):
    n, m = args.n, args.m
    if args.data:
        z = np.load(args.data)
        xk = next((k for k in ("X", "x") if k in z.files), z.files[0])
        x = np.asarray(z[xk][: n + m], dtype=np.float32)
        yk = next((k for k in ("y", "Y", "label") if k in z.files), None)
        yc = (np.asarray(z[yk][: n + m]) > 0.5).astype(np.int64) if yk else None
    else:
        rng = np.random.default_rng(7)
        x = rng.standard_normal((n + m, args.dim)).astype(np.float32)
        yc = None
    x64 = x.astype(np.float64)
    if yc is None:
        yc = (x64[:, 0] + 0.5 * x64[:, 1] - x64[:, 2] * x64[:, 3] > 0).astype(np.int64)
    yr = (x64[:, 0] * 2.0 + x64[:, 1] - x64[:, 2] * x64[:, 3] + np.sin(x64[:, 4])).astype(np.float32)
    y3 = (yc + (x64[:, 5] > 0.8)).astype(np.int64)  # three classes
    x = np.ascontiguousarray(x)
    return x[:n], x[n:], yc[:n], yr[:n], y3[:n]


def timed(fn):
    t = time.perf_counter()
    r = fn()
    return r, time.perf_counter() - t


ALL = ("rf_clf", "rf_reg", "et_clf", "et_reg", "iforest", "gbdt_logloss", "gbdt_rmse",
       "gbdt_depthwise", "gbdt_lossguide", "gbdt_multiclass", "gbdt_ctr", "host_predict_rf",
       "host_predict_gbdt", "dt_clf", "dt_reg", "bagging_clf", "adaboost_clf", "adaboost_reg",
       "dart_reg", "rte", "shap_tree")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="")
    ap.add_argument("--dim", type=int, default=28)
    ap.add_argument("--algos", default="all")
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--m", type=int, default=10_000)
    ap.add_argument("--rows", default="", help="per-algo training row caps, name=N,...")
    ap.add_argument("--trees", type=int, default=20)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import mojolearn as ml

    algos = ALL if args.algos == "all" else tuple(args.algos.split(","))
    caps = dict((k, int(v)) for k, v in (p.split("=") for p in args.rows.split(",") if p))
    rec = dict(threads=os.environ.get("MOJOLEARN_CPU_THREADS", ""), vendor=os.environ.get("MOJOLEARN_VENDOR", ""),
               n=args.n, m=args.m, cells={})
    x, xh, yc, yr, y3 = load(args)
    T = args.trees

    def cell(name, **kv):
        rec["cells"][name] = kv
        print(json.dumps({name: kv}), flush=True)

    def gb(**kw):
        kw.setdefault("learning_rate", 0.03)
        kw.setdefault("random_strength", 0.0)
        kw.setdefault("bootstrap_type", "No")
        return ml.GradientBoosting(**kw)

    def ctr_x(a):
        # two categorical columns above one_hot_max_size (rank groups of
        # columns 3 and 4) in front of the numeric ones
        c0 = np.searchsorted(np.quantile(a[:, 3], [0.3, 0.5, 0.65, 0.77, 0.86, 0.93]), a[:, 3])
        c1 = np.searchsorted(np.quantile(a[:, 4], [0.2, 0.45, 0.7, 0.9]), a[:, 4])
        return np.ascontiguousarray(np.column_stack([c0, c1, a]).astype(np.float32))

    for a in algos:
        n = caps.get(a, args.n)
        X, Yc, Yr, Y3 = x[:n], yc[:n], yr[:n], y3[:n]
        if a in ("rf_clf", "rf_reg", "et_clf", "et_reg"):
            cls = {"rf_clf": ml.RandomForestClassifier, "rf_reg": ml.RandomForestRegressor,
                   "et_clf": ml.ExtraTreesClassifier, "et_reg": ml.ExtraTreesRegressor}[a]
            est = cls(n_estimators=T, max_depth=10, random_state=7)
            _, tf = timed(lambda: est.fit(X, Yc if a.endswith("clf") else Yr))
            p, tp = timed(lambda: est.predict(xh))
            cell(a, n=n, fit_s=round(tf, 3), predict_s=round(tp, 3), out=digest(p))
        elif a == "iforest":
            est = ml.IsolationForest(n_estimators=100, random_state=5)
            _, tf = timed(lambda: est.fit(X))
            s, tp = timed(lambda: est.score_samples(xh))
            cell(a, n=n, fit_s=round(tf, 3), predict_s=round(tp, 3), out=digest(s))
        elif a.startswith("gbdt_"):
            if a == "gbdt_logloss":
                est, yy, xx, xxh = gb(n_estimators=T, max_depth=6, loss="Logloss"), Yc, X, xh
            elif a == "gbdt_rmse":
                est, yy, xx, xxh = gb(n_estimators=T, max_depth=6, loss="RMSE"), Yr, X, xh
            elif a == "gbdt_depthwise":
                est, yy, xx, xxh = gb(n_estimators=T, max_depth=6, grow_policy="Depthwise", loss="Logloss"), Yc, X, xh
            elif a == "gbdt_lossguide":
                est, yy, xx, xxh = gb(n_estimators=T, max_leaves=31, grow_policy="Lossguide", loss="Logloss"), Yc, X, xh
            elif a == "gbdt_multiclass":
                est, yy, xx, xxh = gb(n_estimators=T, max_depth=6, loss="MultiClass"), Y3, X, xh
            elif a == "gbdt_ctr":
                est, yy, xx, xxh = (gb(n_estimators=T, max_depth=6, loss="Logloss", cat_features=[0, 1]), Yc,
                                    ctr_x(X), ctr_x(xh))
            else:
                raise SystemExit(f"unknown algo {a}")
            _, tf = timed(lambda: est.fit(xx, yy))
            p, tp = timed(lambda: est.predict(xxh))
            cell(a, n=n, fit_s=round(tf, 3), predict_s=round(tp, 3), model=digest(est.model_), out=digest(p))
        elif a in ("host_predict_rf", "host_predict_gbdt"):
            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, "m.npz")
                if a == "host_predict_rf":
                    m = ml.RandomForestClassifier(n_estimators=T, max_depth=10, random_state=7).fit(X, Yc)
                else:
                    m = gb(n_estimators=T, max_depth=6, loss="Logloss").fit(X, Yc)
                m.save(path)
                big = np.ascontiguousarray(np.tile(xh, (max(1, n // len(xh)), 1)))
                p, tp = timed(lambda: ml.host_predict(path, big))
            cell(a, n=len(big), predict_s=round(tp, 3), out=digest(p))
        elif a in ("dt_clf", "dt_reg"):
            cls = ml.DecisionTreeClassifier if a == "dt_clf" else ml.DecisionTreeRegressor
            est = cls(max_depth=12, random_state=7)
            _, tf = timed(lambda: est.fit(X, Yc if a == "dt_clf" else Yr))
            p, tp = timed(lambda: est.predict(xh))
            cell(a, n=n, fit_s=round(tf, 3), predict_s=round(tp, 3), out=digest(p))
        elif a == "bagging_clf":
            est = ml.BaggingClassifier(ml.DecisionTreeClassifier(max_depth=8), n_estimators=10, max_samples=0.8,
                                       max_features=0.75, random_state=7)
            _, tf = timed(lambda: est.fit(X, Yc))
            p, tp = timed(lambda: est.predict_proba(xh))
            cell(a, n=n, fit_s=round(tf, 3), predict_s=round(tp, 3), out=digest(p))
        elif a in ("adaboost_clf", "adaboost_reg"):
            if a == "adaboost_clf":
                est = ml.AdaBoostClassifier(ml.DecisionTreeClassifier(max_depth=2), n_estimators=T, random_state=7)
                yy = Yc
            else:
                est = ml.AdaBoostRegressor(ml.DecisionTreeRegressor(max_depth=3), n_estimators=T, random_state=7)
                yy = Yr
            _, tf = timed(lambda: est.fit(X, yy))
            p, tp = timed(lambda: est.predict(xh))
            cell(a, n=n, fit_s=round(tf, 3), predict_s=round(tp, 3), out=digest(p))
        elif a == "dart_reg":
            est = ml.DARTRegressor(n_estimators=T, num_leaves=31, max_depth=6, min_child_samples=20, drop_rate=0.1,
                                   skip_drop=0.5, drop_seed=3, random_state=7)
            _, tf = timed(lambda: est.fit(X, Yr))
            p, tp = timed(lambda: est.predict(xh))
            cell(a, n=n, fit_s=round(tf, 3), predict_s=round(tp, 3), out=digest(p))
        elif a == "rte":
            est = ml.RandomTreesEmbedding(n_estimators=T, max_depth=5, random_state=7)
            _, tf = timed(lambda: est.fit(X))
            p, tp = timed(lambda: est.transform(xh))
            cell(a, n=n, fit_s=round(tf, 3), predict_s=round(tp, 3), out=digest(p))
        elif a == "shap_tree":
            dart = ml.DARTRegressor(n_estimators=T, num_leaves=15, max_depth=5, random_state=7).fit(X, Yr)
            e = ml.TreeExplainer(dart, data=X[:256])
            s, ts = timed(lambda: e.shap_values(xh[:1000]))
            cell(a, n=n, shap_s=round(ts, 3), out=digest(s))
        else:
            raise SystemExit(f"unknown algo {a}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rec, f, indent=1)


if __name__ == "__main__":
    main()
