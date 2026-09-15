#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The learning-to-rank arm of the GBDT benchmark: Istella-S with query ids.

One process per (library, loss), so a peak GPU memory reading belongs to one
cell and a crash costs one cell:

    python3 tools/speed_gbdt_rank.py --library ours     --loss QueryRMSE
    python3 tools/speed_gbdt_rank.py --library catboost --loss YetiRank
    python3 tools/speed_gbdt_rank.py --library xgboost  --loss rank:ndcg
    python3 tools/speed_gbdt_rank.py --library lightgbm --loss lambdarank --device cpu
    python3 tools/speed_gbdt_rank.py --selftest

`tools/speed_gbdt_arm.py` keeps its binary and regression Istella cells
unchanged; this file imports its loader (`load_istella_rank`) and nothing of
its CLI.

THE CONFIG, HELD EQUAL WHERE THE LIBRARIES ALLOW (`params_for`), every
mismatch named there and in the results README: 100 trees, depth 6, learning
rate 0.1, L2 1.0, 254 borders (255 bins), no row or column sampling, seed 7.

THE CLOCK. Each timed fit is construction + the library's own input
conversion (CatBoost `Pool`, XGBoost's `DMatrix` inside `fit`, LightGBM's
`Dataset`, our `group_id` run lengths) + fit + a device drain
(`cudaDeviceSynchronize` through libcudart, the same call after every arm).
One untimed 1-tree warm-up fit per process first. Tree counts 1, 10 and 100,
each `--repeats` times, the order rotated per repeat. The per-tree cost is
the least-squares slope through the three medians and the fixed cost its
intercept: what an arm pays before it boosts at all.

QUALITY. NDCG@5 and NDCG@10 on the whole Istella-S test split, computed HERE
from each library's raw test scores by one function (`ndcg_at`): gain
2^grade - 1, discount 1/log2(rank + 1), a query with no relevant document
scores 1.0 (LightGBM's and CatBoost's convention), mean over queries. Ties in
the predicted score are broken PESSIMISTICALLY (lower grade first), so no arm
is credited for the file order; `ndcg*_fileorder` breaks them by row order
and shows how much ties matter.

MEMORY. `nvidia-smi --query-gpu=memory.used` sampled every 100 ms for the
whole process; peak minus the reading taken before any library is imported.

Output: one JSON per process (`--json`), rewritten after every fit so a
timeout still leaves every finished fit; `RANK ` lines on stdout.
"""

import argparse
import ctypes
import glob
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, os.path.join(_ROOT, "python")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

LOSSES = {
    "ours": ("QueryRMSE", "PairLogit", "YetiRank"),
    "catboost": ("QueryRMSE", "PairLogit", "YetiRank"),
    "xgboost": ("rank:pairwise", "rank:ndcg"),
    "lightgbm": ("lambdarank",),
}

N_TREES = 100
DEPTH = 6
LR = 0.1
L2 = 1.0
BORDERS = 254
SEED = 7


# --------------------------------------------------------------------------
# NDCG, one implementation for every library.
# --------------------------------------------------------------------------

def query_starts(qid):
    change = np.flatnonzero(np.diff(qid) != 0) + 1
    return np.concatenate([[0], change, [qid.size]])


def ndcg_at(k, bounds, grades, scores, pessimistic=True):
    gains_all = np.power(2.0, grades.astype(np.float64)) - 1.0
    disc = 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float64))
    total = 0.0
    n_q = bounds.size - 1
    for i in range(n_q):
        a, b = bounds[i], bounds[i + 1]
        g = gains_all[a:b]
        s = scores[a:b].astype(np.float64)
        if pessimistic:
            order = np.lexsort((g, -s))          # score desc, then grade asc
        else:
            order = np.argsort(-s, kind="stable")
        kk = min(k, b - a)
        dcg = float(np.dot(g[order[:kk]], disc[:kk]))
        ideal = np.sort(g)[::-1][:kk]
        idcg = float(np.dot(ideal, disc[:kk]))
        total += 1.0 if idcg == 0.0 else dcg / idcg
    return total / n_q


def selftest():
    qid = np.array([1, 1, 1, 2, 2])
    y = np.array([3, 0, 1, 0, 0], dtype=np.float32)
    b = query_starts(qid)
    perfect = np.array([3.0, 1.0, 2.0, 0.0, 0.0])
    assert abs(ndcg_at(10, b, y, perfect) - 1.0) < 1e-12
    worst = -perfect
    want = (1.0 / np.log2(4) * 7 + 1.0 / np.log2(3) * 1) / (7 + 1 / np.log2(3))
    got = ndcg_at(10, b, y, worst)
    assert abs(got - (want + 1.0) / 2) < 1e-12, got
    ties = np.zeros(5)
    assert ndcg_at(10, b, y, ties) == ndcg_at(10, b, y, worst)
    fileorder = (7.0 + 1.0 / np.log2(4)) / (7 + 1 / np.log2(3))
    got = ndcg_at(10, b, y, ties, pessimistic=False)
    assert abs(got - (fileorder + 1.0) / 2) < 1e-12, got
    print("RANK selftest OK")


# --------------------------------------------------------------------------
# Device drain and memory sampler, identical for every arm.
# --------------------------------------------------------------------------

def make_drain():
    names = ["libcudart.so", "libcudart.so.13", "libcudart.so.12"]
    names += glob.glob("/usr/local/cuda/lib64/libcudart.so*")
    try:
        import importlib.util
        spec = importlib.util.find_spec("torch")
        if spec and spec.origin:
            names += glob.glob(os.path.join(os.path.dirname(spec.origin),
                                            "lib", "libcudart*"))
    except Exception:                               # noqa: BLE001
        pass
    for name in names:
        try:
            lib = ctypes.CDLL(name)
            fn = lib.cudaDeviceSynchronize
            fn.restype = ctypes.c_int
            return name, fn
        except (OSError, AttributeError):
            continue
    return None, (lambda: 0)


class MemSampler(object):
    def __init__(self):
        self.samples = []
        self.proc = None

    def read_once(self):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=20).stdout
            return int(out.split()[0])
        except Exception:                           # noqa: BLE001
            return None

    def start(self):
        try:
            self.proc = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits", "-lms", "100"],
                stdout=subprocess.PIPE, text=True)
        except OSError:
            return
        def pump():
            for line in self.proc.stdout:
                try:
                    self.samples.append(int(line.split()[0]))
                except (ValueError, IndexError):
                    pass
        threading.Thread(target=pump, daemon=True).start()

    def peak(self):
        return max(self.samples) if self.samples else None

    def stop(self):
        if self.proc:
            self.proc.terminate()


# --------------------------------------------------------------------------
# The arms.
# --------------------------------------------------------------------------

def params_for(library, loss, device):
    """The knobs, and every place a library could not be matched."""
    notes = []
    if library == "ours":
        p = dict(loss=loss, max_depth=DEPTH, learning_rate=LR,
                 l2_leaf_reg=L2, border_count=BORDERS, random_state=SEED,
                 bootstrap_type="No", grow_policy="SymmetricTree")
        notes.append("SymmetricTree (the ranking losses fit on the symmetric "
                     "searcher only)")
        if loss == "YetiRank":
            notes.append("l2_leaf_reg pinned 1.0 (the reference default is 0)")
    elif library == "catboost":
        p = dict(loss_function=loss, depth=DEPTH, learning_rate=LR,
                 l2_leaf_reg=L2, border_count=BORDERS, random_seed=SEED,
                 bootstrap_type="No", boosting_type="Plain",
                 grow_policy="SymmetricTree", task_type=device.upper(),
                 verbose=False, allow_writing_files=False)
        if device == "gpu":
            p["devices"] = "0"
        if loss == "YetiRank":
            notes.append("l2_leaf_reg pinned 1.0 (its default is 0)")
    elif library == "xgboost":
        p = dict(objective=loss, tree_method="hist",
                 device="cuda" if device == "gpu" else "cpu",
                 max_depth=DEPTH, learning_rate=LR, reg_lambda=L2,
                 max_bin=BORDERS + 1, subsample=1.0, colsample_bytree=1.0,
                 random_state=SEED)
        notes.append("depthwise trees, not symmetric (XGBoost has no "
                     "oblivious grower)")
        notes.append("pair construction left at XGBoost's defaults "
                     "(lambdarank_pair_method, lambdarank_num_pair_per_sample)")
        notes.append("min_child_weight 1 (hessian), not a row count")
    elif library == "lightgbm":
        p = dict(objective=loss, max_depth=DEPTH, num_leaves=2 ** DEPTH,
                 learning_rate=LR, reg_lambda=L2, max_bin=BORDERS + 1,
                 subsample=1.0, subsample_freq=0, colsample_bytree=1.0,
                 random_state=SEED, verbose=-1,
                 device_type="cuda" if device == "gpu" else "cpu")
        notes.append("leaf-wise growth capped at depth 6 and 64 leaves, not "
                     "symmetric")
        notes.append("min_child_samples 20 and min_sum_hessian 1e-3 left at "
                     "LightGBM's defaults; lambdarank_truncation_level 30 "
                     "default")
    else:
        raise SystemExit("unknown library " + library)
    return p, notes


def version_of(library):
    if library == "ours":
        import mojolearn
        return getattr(mojolearn, "__version__", "source")
    return __import__(library).__version__


def build_arm(library, loss, device, data):
    params, notes = params_for(library, loss, device)
    x, y, qid = data.x_train, data.r_train, data.qid_train
    sizes = np.diff(query_starts(qid))

    if library == "ours":
        import mojolearn

        def fit(n):
            m = mojolearn.GradientBoosting(n_estimators=n, **params)
            m.fit(x, y, group_id=qid)
            return m

        def predict(m):
            return np.asarray(m.predict(data.x_test), dtype=np.float64)

    elif library == "catboost":
        import catboost

        def fit(n):
            pool = catboost.Pool(x, y, group_id=qid)
            m = catboost.CatBoost(dict(params, iterations=n))
            m.fit(pool)
            return m

        def predict(m):
            return np.asarray(m.predict(data.x_test,
                                        prediction_type="RawFormulaVal"),
                              dtype=np.float64)

    elif library == "xgboost":
        import xgboost
        # XGBoost REFUSES a qid that is not sorted non-decreasing
        # (`data.cc:622`), and Istella-S numbers its queries in neither file
        # in ascending order (the test split opens at qid 032047). Ours and
        # CatBoost ask only that a query's rows be CONSECUTIVE. So each query
        # is relabeled by its order of appearance: the same partition of the
        # same rows in the same order, with names XGBoost accepts. No row
        # moves, so no arm trains on different data.
        qid_x = np.repeat(np.arange(sizes.size, dtype=np.uint32), sizes)

        def fit(n):
            m = xgboost.XGBRanker(n_estimators=n, **params)
            m.fit(x, y, qid=qid_x)
            return m

        def predict(m):
            return np.asarray(m.predict(data.x_test), dtype=np.float64)

    else:
        import lightgbm

        def fit(n):
            m = lightgbm.LGBMRanker(n_estimators=n, **params)
            m.fit(x, y, group=sizes)
            return m

        def predict(m):
            return np.asarray(m.predict(data.x_test), dtype=np.float64)

    return params, notes, fit, predict


def fit_line(counts, medians):
    a = np.vstack([np.asarray(counts, dtype=np.float64),
                   np.ones(len(counts))]).T
    slope, intercept = np.linalg.lstsq(a, np.asarray(medians), rcond=None)[0]
    return float(intercept), float(slope)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--library", choices=sorted(LOSSES))
    ap.add_argument("--loss")
    ap.add_argument("--device", default="gpu", choices=("gpu", "cpu"))
    ap.add_argument("--trees", default="1,10,100")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--rows", type=int, default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
        return
    if args.loss not in LOSSES.get(args.library, ()):
        raise SystemExit("loss %r is not run for %s (%s)"
                         % (args.loss, args.library, LOSSES.get(args.library)))
    if args.library == "ours" and args.device != "gpu":
        raise SystemExit("ours runs on the GPU only in this arm")

    sampler = MemSampler()
    baseline = sampler.read_once()
    sampler.start()
    drain_name, drain = make_drain()

    import speed_gbdt_arm as spec
    t0 = time.perf_counter()
    data = spec.load_istella_rank(args.rows)
    load_s = time.perf_counter() - t0
    bounds = query_starts(data.qid_test)
    counts = [int(c) for c in args.trees.split(",")]

    params, notes, fit, predict = build_arm(args.library, args.loss,
                                            args.device, data)
    rec = dict(
        library=args.library, loss=args.loss, device=args.device,
        version=version_of(args.library), params=params, mismatches=notes,
        numeric_mode=os.environ.get("MOJOLEARN_NUMERIC_MODE", ""),
        rows_train=int(data.x_train.shape[0]),
        queries_train=int(query_starts(data.qid_train).size - 1),
        rows_test=int(data.x_test.shape[0]), queries_test=int(bounds.size - 1),
        features=int(data.x_train.shape[1]), load_s=load_s,
        drain=drain_name, python=platform.python_version(),
        mem_baseline_mib=baseline, fits=[], quality=[])
    print("RANK start %s %s %s rows=%d queries=%d version=%s drain=%s"
          % (args.library, args.loss, args.device, rec["rows_train"],
             rec["queries_train"], rec["version"], drain_name))

    def dump():
        rec["mem_peak_mib"] = sampler.peak()
        if args.json:
            tmp = args.json + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(rec, fh, indent=1, sort_keys=True)
            os.replace(tmp, args.json)

    if args.library == "ours":
        from mojolearn.ensemble import _group_sizes
        t0 = time.perf_counter()
        _group_sizes(data.qid_train, data.qid_train.size)
        rec["ours_group_sizes_ms"] = (time.perf_counter() - t0) * 1e3
        import mojolearn
        probe = mojolearn.GradientBoosting()
        rec["ours_mode_used"] = str(probe.numeric_mode_used())
        rec["ours_vendor_used"] = str(probe.vendor_used())

    t0 = time.perf_counter()
    fit(1)
    drain()
    rec["warmup_ms"] = (time.perf_counter() - t0) * 1e3
    print("RANK warmup %.1f ms" % rec["warmup_ms"])
    dump()

    for rep in range(args.repeats):
        order = counts[rep % len(counts):] + counts[:rep % len(counts)]
        for n in order:
            t0 = time.perf_counter()
            m = fit(n)
            drain()
            ms = (time.perf_counter() - t0) * 1e3
            rec["fits"].append(dict(repeat=rep, trees=n, ms=ms))
            print("RANK fit rep=%d trees=%d ms=%.1f" % (rep, n, ms))
            if n == max(counts):
                s = predict(m)
                q = dict(repeat=rep, trees=n,
                         ndcg5=ndcg_at(5, bounds, data.r_test, s),
                         ndcg10=ndcg_at(10, bounds, data.r_test, s),
                         ndcg5_fileorder=ndcg_at(5, bounds, data.r_test, s,
                                                 False),
                         ndcg10_fileorder=ndcg_at(10, bounds, data.r_test, s,
                                                  False),
                         pred_sha=hashlib.sha256(
                             s.astype(np.float64).tobytes()).hexdigest()[:16])
                rec["quality"].append(q)
                print("RANK quality rep=%d ndcg5=%.6f ndcg10=%.6f hash=%s"
                      % (rep, q["ndcg5"], q["ndcg10"], q["pred_sha"]))
            del m
            dump()

    summary = {}
    for n in counts:
        ts = [f["ms"] for f in rec["fits"] if f["trees"] == n]
        summary[str(n)] = dict(median=statistics.median(ts), min=min(ts),
                               max=max(ts), n=len(ts))
    fixed, per_tree = fit_line(counts, [summary[str(n)]["median"]
                                        for n in counts])
    rec["summary"] = summary
    rec["fixed_ms"] = fixed
    rec["per_tree_ms"] = per_tree
    sampler.stop()
    dump()
    print("RANK done %s %s median@%d=%.1f fixed=%.1f per_tree=%.3f peak=%s base=%s"
          % (args.library, args.loss, max(counts),
             summary[str(max(counts))]["median"], fixed, per_tree,
             rec["mem_peak_mib"], baseline))


if __name__ == "__main__":
    main()
