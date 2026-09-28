#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Lossguide quality of ONE build, paired by subset across builds: for each
seed (a different 300k training subset, the same 200k test rows) fit 300
trees with max_leaves 31 and max_depth 10, where the leaf budget binds, and
print the test metrics. The method of
bench/results/fast_quality_audit_2026-09-26/lg_quality.py, with the datasets
and seeds named on the command line.

    pixi run -e default python bench/speed/trees_apple3_lg_quality.py <tag> taxi,istellareg 5

Every line begins `LGQ ` and is one JSON record."""
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.environ.get("TAP_REPO") or os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (os.path.join(_ROOT, "tools"), os.path.join(_ROOT, "python"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import speed_gbdt_arm as spec  # noqa: E402
import mojolearn as M  # noqa: E402


def main(argv):
    tag = argv[1]
    datasets = argv[2].split(",") if len(argv) > 2 else ["taxi", "istellareg"]
    seeds = int(argv[3]) if len(argv) > 3 else 5
    trees = int(os.environ.get("LGQ_TREES", "300"))
    for ds in datasets:
        d = spec.load_with_fallback(ds, "shipped", 2_000_000)
        if not d.tag.startswith(ds.replace("reg", "")) and not d.tag.startswith(ds):
            raise SystemExit("LGQ FATAL: asked for %r, the loader produced %r" % (ds, d.tag))
        x_all = np.asarray(d.X_train, dtype=np.float32)
        y_all = np.asarray(d.y_train)
        xt = np.ascontiguousarray(np.asarray(d.X_test, dtype=np.float32)[:200000])
        yt = np.asarray(d.y_test)[:200000]
        for seed in range(seeds):
            rng = np.random.default_rng(1000 + seed)
            idx = np.sort(rng.choice(len(x_all), size=min(300000, len(x_all)), replace=False))
            x = np.ascontiguousarray(x_all[idx])
            y = y_all[idx]
            kw = dict(n_estimators=trees, grow_policy="Lossguide", max_leaves=31, max_depth=10,
                      random_state=seed)
            t = time.time()
            if d.task != "regression":
                m = M.GradientBoostingClassifier(**kw).fit(x, y.astype(np.int64))
                fit_s = time.time() - t
                p = np.asarray(m.predict_proba(xt))[:, 1].astype(np.float64)
                eps = 1e-12
                ll = -np.mean(yt * np.log(p + eps) + (1 - yt) * np.log(1 - p + eps))
                acc = np.mean((p > 0.5) == (yt == 1))
                order = np.argsort(p)
                r = np.empty(len(p))
                r[order] = np.arange(len(p))
                pos = yt == 1
                auc = (r[pos].sum() - pos.sum() * (pos.sum() - 1) / 2) / (pos.sum() * (~pos).sum())
                rec = dict(ds=ds, seed=seed, logloss=float(ll), acc=float(acc), auc=float(auc))
            else:
                m = M.GradientBoostingRegressor(**kw).fit(x, y.astype(np.float32))
                fit_s = time.time() - t
                p = np.asarray(m.predict(xt), dtype=np.float64)
                rec = dict(ds=ds, seed=seed, rmse=float(np.sqrt(np.mean((p - yt) ** 2))))
            rec["digest"] = spec.hash_predictions(np.asarray(p, dtype=np.float64))
            rec["fit_s"] = round(fit_s, 3)
            rec["tag"] = tag
            rec["mode"] = os.environ.get("MOJOLEARN_NUMERIC_MODE", "unset")
            print("LGQ " + json.dumps(rec, sort_keys=True), flush=True)


if __name__ == "__main__":
    main(sys.argv)
