#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Inference timing of the classical estimators, arm against arm, on the two
benchmark datasets (lane/infer-speed-classical, 2026-09-17).

`tools/classical_two_datasets.py` times FIT for ols and pca and the public
kneighbors and score_samples calls for knn and kde, one library against
another. This file times INFERENCE ONLY, one build of this library against
another build of it, on the blocks that tool's `prep` wrote:

    python3 bench/speed/classical_ladder_infer.py fit --data DIR --models DIR \\
        [--lanes ols,pca,knn,kde] [--datasets taxi,istella]
    python3 bench/speed/classical_ladder_infer.py race --data DIR --models DIR \\
        --arms arms.json --out DIR [--lanes ...] [--datasets ...] \\
        [--outer 5] [--rounds 3] [--warmup 1] [--rows-cpu-knn 400] [--rows-cpu-kde 200]
    python3 bench/speed/classical_ladder_infer.py worker ...   (spawned by race)

fit   fits each (lane, dataset) ONCE on the GPU build the caller's
      environment selects and saves the model (`save`), so every arm
      predicts from the SAME fitted bytes: train on the GPU, infer anywhere.
race  for `--outer` rounds, spawns one worker PROCESS per arm in an order
      rotated every round; a worker loads the arm's build (its `env` from
      arms.json, at least PYTHONPATH and, for a CPU arm, MOJOLEARN_HOST_DIR),
      loads the block and the saved model, runs `--warmup` untimed calls and
      `--rounds` timed calls of the public inference method (predict,
      transform, kneighbors, score_samples; host array in, host array out,
      the upload inside the clock as the public call uploads), and reports
      every call's milliseconds and the SHA-256 of its output bytes. The
      conductor requires every digest of a (lane, dataset) to be equal
      across arms, rounds and calls, computes each arm's median, min, max and
      spread (max over min), and the paired ratio of each arm to the first
      arm (the median over outer rounds of the per-round medians' ratio).
      A cell whose spread exceeds 1.10 is reported and marked, never quoted.

arms.json: {"<arm name>": {"python": "<interpreter>", "cwd": "<dir>",
            "env": {"NAME": "value", ...}, "cpu": true|false}}. A CPU arm's
kNN and KDE cells use the first `--rows-cpu-knn` queries and the first
`--rows-cpu-kde` queries of the block (the index and the training set stay
whole), because the serial CPU walk at the full 4,000 and 2,000 queries
takes minutes per call; the GPU arms use the whole block. Every arm of one
cell sees the same rows, so a ratio inside a cell is like for like; a ratio
ACROSS a GPU cell and a CPU cell is not, and none is printed.

Output: <out>/<lane>-<dataset>.json per cell and <out>/summary.tsv.
"""
import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time

import numpy as np

LANES = ("ols", "pca", "knn", "kde")
DATASETS = ("taxi", "istella")
BLOCK_OF = {"ols": "big", "pca": "big", "knn": "knn", "kde": "kde"}
KNN_K = 10
PCA_COMPONENTS = 8


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def load_block(data, lane, dataset):
    base = os.path.join(data, "%s-%s" % (BLOCK_OF[lane], dataset))
    with np.load(base + ".npz") as z:
        arrays = {k: np.ascontiguousarray(z[k]) for k in z.files}
    with open(base + ".json") as fh:
        rec = json.load(fh)
    return arrays, rec


def model_path(models, lane, dataset):
    return os.path.join(models, "%s-%s.npz" % (lane, dataset))


def cmd_fit(args):
    import mojolearn as ml
    os.makedirs(args.models, exist_ok=True)
    facts = {"vendor": ml.vendor(), "numeric_mode": ml.numeric_mode(), "models": {}}
    for lane in args.lanes.split(","):
        for ds in args.datasets.split(","):
            arrays, rec = load_block(args.data, lane, ds)
            t0 = time.perf_counter()
            if lane == "ols":
                m = ml.LinearRegression(fit_intercept=True).fit(arrays["X"], arrays["y"])
            elif lane == "pca":
                m = ml.PCA(n_components=PCA_COMPONENTS, svd_solver="covariance_eigh").fit(arrays["X"])
            elif lane == "knn":
                m = ml.NearestNeighbors(n_neighbors=KNN_K).fit(arrays["index"])
            elif lane == "kde":
                m = ml.KernelDensity(bandwidth=rec["kde"]["bandwidth"], kernel="gaussian").fit(arrays["X"])
            else:
                raise SystemExit("unknown lane %r" % lane)
            fit_ms = (time.perf_counter() - t0) * 1000.0
            path = model_path(args.models, lane, ds)
            m.save(path)
            with open(path, "rb") as fh:
                digest = sha256_bytes(fh.read())
            facts["models"]["%s-%s" % (lane, ds)] = {
                "path": path, "sha256": digest, "fit_ms": fit_ms,
                "fit_rows": int(arrays["X"].shape[0]) if "X" in arrays else int(arrays["index"].shape[0]),
                "numeric_mode_used": m.numeric_mode_used(), "vendor_used": m.vendor_used()}
            print("FIT %s-%s %.1f ms -> %s" % (lane, ds, fit_ms, path), flush=True)
    with open(os.path.join(args.models, "fit.json"), "w") as fh:
        json.dump(facts, fh, indent=1, sort_keys=True)
    return 0


def _load_model(ml, lane, path):
    if ml.vendor() == "cpu":
        from mojolearn._classical_host import host_model
        return host_model(path)
    cls = {"ols": ml.LinearRegression, "pca": ml.PCA, "knn": ml.NearestNeighbors,
           "kde": ml.KernelDensity}[lane]
    return cls.load(path)


def cmd_worker(args):
    import mojolearn as ml
    arrays, rec = load_block(args.data, args.lane, args.dataset)
    m = _load_model(ml, args.lane, model_path(args.models, args.lane, args.dataset))
    if args.lane in ("ols", "pca"):
        xin = arrays["Xq"]
    elif args.lane == "knn":
        xin = arrays["queries"]
    else:
        xin = arrays["Xq"]
    if args.rows and args.rows < xin.shape[0]:
        xin = np.ascontiguousarray(xin[:args.rows])
    call = {"ols": lambda: m.predict(xin), "pca": lambda: m.transform(xin),
            "knn": lambda: m.kneighbors(xin), "kde": lambda: m.score_samples(xin)}[args.lane]

    def digest(out):
        if isinstance(out, tuple):
            h = hashlib.sha256()
            for part in out:
                h.update(np.ascontiguousarray(part).tobytes())
            return h.hexdigest()
        return sha256_bytes(np.ascontiguousarray(out).tobytes())

    for _ in range(args.warmup):
        call()
    ms, digests = [], []
    for _ in range(args.rounds):
        t0 = time.perf_counter()
        out = call()
        ms.append((time.perf_counter() - t0) * 1000.0)
        digests.append(digest(out))
    result = {"arm": args.arm, "lane": args.lane, "dataset": args.dataset,
              "rows": int(xin.shape[0]), "features": int(xin.shape[1]),
              "vendor": ml.vendor(), "numeric_mode": ml.numeric_mode(),
              "numeric_mode_used": m.numeric_mode_used(), "vendor_used": m.vendor_used(),
              "pid": os.getpid(), "ms": ms, "digests": digests,
              "env": {k: os.environ.get(k) for k in ("PYTHONPATH", "MOJOLEARN_HOST_DIR",
                                                     "MOJOLEARN_CPU_THREADS", "MOJOLEARN_NUMERIC_MODE")}}
    if hasattr(m, "model_sha256"):
        try:
            result["model_sha256"] = m.model_sha256()
        except Exception as exc:  # noqa: BLE001
            result["model_sha256"] = "unavailable (%r)" % (exc,)
    with open(args.json, "w") as fh:
        json.dump(result, fh, indent=1, sort_keys=True)
    print("WORKER %s %s-%s rows=%d median=%.3f ms digest=%s" % (
        args.arm, args.lane, args.dataset, xin.shape[0], statistics.median(ms), digests[0][:16]), flush=True)
    return 0


def cmd_race(args):
    with open(args.arms) as fh:
        arms = json.load(fh)
    names = list(arms)
    os.makedirs(args.out, exist_ok=True)
    here = os.path.abspath(__file__)
    summary_rows = []
    for lane in args.lanes.split(","):
        for ds in args.datasets.split(","):
            cell = {"lane": lane, "dataset": ds, "outer": args.outer, "rounds": args.rounds,
                    "warmup": args.warmup, "arms": {n: {"runs": []} for n in names},
                    "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            for r in range(args.outer):
                order = names[r % len(names):] + names[:r % len(names)]
                for arm in order:
                    spec = arms[arm]
                    rows = 0
                    if spec.get("cpu"):
                        rows = {"knn": args.rows_cpu_knn, "kde": args.rows_cpu_kde}.get(lane, 0)
                    jpath = os.path.join(args.out, "%s-%s-%s-r%d.json" % (lane, ds, arm, r))
                    env = dict(os.environ)
                    env.update(spec.get("env", {}))
                    cmd = [spec["python"], here, "worker", "--arm", arm, "--lane", lane,
                           "--dataset", ds, "--data", args.data, "--models", args.models,
                           "--rounds", str(args.rounds), "--warmup", str(args.warmup),
                           "--rows", str(rows), "--json", jpath]
                    log = os.path.join(args.out, "%s-%s-%s-r%d.log" % (lane, ds, arm, r))
                    with open(log, "w") as lf:
                        rc = subprocess.call(cmd, env=env, cwd=spec.get("cwd"), stdout=lf, stderr=subprocess.STDOUT)
                    if rc != 0 or not os.path.exists(jpath):
                        cell["arms"][arm]["runs"].append({"round": r, "error": "exit %d, see %s" % (rc, log)})
                        print("ERROR %s-%s %s round %d exit %d" % (lane, ds, arm, r, rc), flush=True)
                        continue
                    with open(jpath) as fh:
                        run = json.load(fh)
                    run["round"] = r
                    cell["arms"][arm]["runs"].append(run)
                    print("ROUND %d %s-%s %s rows=%d median=%.3f ms" % (
                        r, lane, ds, arm, run["rows"], statistics.median(run["ms"])), flush=True)
            # aggregate
            digests_by_rows = {}
            for arm in names:
                runs = [x for x in cell["arms"][arm]["runs"] if "ms" in x]
                a = cell["arms"][arm]
                a["ok_rounds"] = len(runs)
                if not runs:
                    a["status"] = "failed"
                    continue
                allms = [v for x in runs for v in x["ms"]]
                a["median_ms"] = statistics.median(allms)
                a["min_ms"] = min(allms)
                a["max_ms"] = max(allms)
                a["spread"] = a["max_ms"] / a["min_ms"] if a["min_ms"] > 0 else None
                a["round_medians"] = [statistics.median(x["ms"]) for x in runs]
                a["rows"] = runs[0]["rows"]
                a["vendor"] = runs[0]["vendor"]
                ds_set = {d for x in runs for d in x["digests"]}
                a["digests"] = sorted(ds_set)
                a["digest_stable"] = len(ds_set) == 1
                a["status"] = "ok" if len(runs) == args.outer and a["digest_stable"] and a["spread"] is not None and a["spread"] <= 1.10 else "flagged"
                digests_by_rows.setdefault(a["rows"], set()).update(ds_set)
            cell["digest_equal_across_arms_per_rows"] = {str(k): len(v) == 1 for k, v in digests_by_rows.items()}
            first = names[0]
            ratios = {}
            for arm in names[1:]:
                a, b = cell["arms"][first], cell["arms"][arm]
                if "round_medians" in a and "round_medians" in b and a.get("rows") == b.get("rows"):
                    pairs = [y / x for x, y in zip(a["round_medians"], b["round_medians"]) if x > 0]
                    ratios[arm] = {"over": first, "median_ratio": b["median_ms"] / a["median_ms"],
                                   "paired_ratio": statistics.median(pairs) if pairs else None,
                                   "paired": pairs}
            cell["ratios"] = ratios
            cell["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            with open(os.path.join(args.out, "%s-%s.json" % (lane, ds)), "w") as fh:
                json.dump(cell, fh, indent=1, sort_keys=True)
            for arm in names:
                a = cell["arms"][arm]
                summary_rows.append((lane, ds, arm, a.get("rows"), a.get("ok_rounds"), a.get("median_ms"),
                                     a.get("min_ms"), a.get("max_ms"), a.get("spread"), a.get("digest_stable"),
                                     a.get("status"), (ratios.get(arm) or {}).get("paired_ratio")))
            print("CELL %s-%s " % (lane, ds) + " ".join(
                "%s=%.3fms" % (n, cell["arms"][n]["median_ms"]) for n in names if "median_ms" in cell["arms"][n])
                + " ratios=" + json.dumps({k: round(v["paired_ratio"], 4) if v["paired_ratio"] else None for k, v in ratios.items()})
                + " digests_equal=" + json.dumps(cell["digest_equal_across_arms_per_rows"]), flush=True)
    with open(os.path.join(args.out, "summary.tsv"), "w") as fh:
        fh.write("lane\tdataset\tarm\trows\tok_rounds\tmedian_ms\tmin_ms\tmax_ms\tspread\tdigest_stable\tstatus\tpaired_ratio_over_first\n")
        for row in summary_rows:
            fh.write("\t".join("" if v is None else ("%.4f" % v if isinstance(v, float) else str(v)) for v in row) + "\n")
    return 0


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--data", required=True)
    f.add_argument("--models", required=True)
    f.add_argument("--lanes", default=",".join(LANES))
    f.add_argument("--datasets", default=",".join(DATASETS))
    w = sub.add_parser("worker")
    w.add_argument("--arm", required=True)
    w.add_argument("--lane", required=True, choices=LANES)
    w.add_argument("--dataset", required=True, choices=DATASETS)
    w.add_argument("--data", required=True)
    w.add_argument("--models", required=True)
    w.add_argument("--rounds", type=int, default=3)
    w.add_argument("--warmup", type=int, default=1)
    w.add_argument("--rows", type=int, default=0)
    w.add_argument("--json", required=True)
    r = sub.add_parser("race")
    r.add_argument("--data", required=True)
    r.add_argument("--models", required=True)
    r.add_argument("--arms", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--lanes", default=",".join(LANES))
    r.add_argument("--datasets", default=",".join(DATASETS))
    r.add_argument("--outer", type=int, default=5)
    r.add_argument("--rounds", type=int, default=3)
    r.add_argument("--warmup", type=int, default=1)
    r.add_argument("--rows-cpu-knn", type=int, default=400)
    r.add_argument("--rows-cpu-kde", type=int, default=200)
    args = p.parse_args()
    return {"fit": cmd_fit, "worker": cmd_worker, "race": cmd_race}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
