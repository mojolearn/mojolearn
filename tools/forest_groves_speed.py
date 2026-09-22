#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Timing of the `parallel_groves` engine, one path per process
(lane/forest-groves-cpu-and-speed, 2026-09-17), the shape of
bench/speed/infer_speed_trees_ab.py: `prepare` trains each forest ONCE on the
box's GPU and saves it as a `-parallel-groves-1` archive beside its fixed
prediction rows; `time` loads one archive in the checkout `PYTHONPATH` names
and predicts the fixed rows through ONE path; `summarize` pairs the arms.

    prepare   --out DIR [--higgs-train 1000000] [--models ...]
              rf-higgs-100x16, et-higgs-100x16, rf-covtype-100x16, et-year-100x16,
              rf-higgs-500x16 (100 trees depth 16 and 500 trees depth 16), the
              same constructor the speed harness's `ours` arm uses
              (bench/speed/forest_speed_arm.py); the prediction rows are HIGGS's
              500,000 held-out rows, every Covtype row (581,012) and every Year
              row (515,345), float32 C-order .npy; the training rows and labels
              are saved too for the cuML arm (tools/forest_groves_fil.py)
    time      --model X.npz --x rows.npy --path {groves,sequential,host-groves}
              groves: the GPU class with inference_engine='parallel_groves' (what
              the archive restores); sequential: the same fit under the default
              engine, for context; host-groves: mojolearn.host_model(<archive>),
              the CPU door. One warmup, then --rounds single calls, then
              --rounds blocks of --calls calls (throughput, ms per call). Every
              output is hashed; the loaded .so files are digested from inside
              the process.
    summarize JSON... [--before before --after after]

The gate is at least five timed
calls after a warmup, a max/min spread of at most 1.10 per arm, output hashes
equal across arms and rounds. A process outside the gate is written with
`stable: false`, and `summarize` quotes no ratio from it.
"""
import argparse
import glob
import hashlib
import json
import os
import statistics
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "bench", "speed"))
sys.path.insert(0, os.path.join(ROOT, "python"))

SPREAD_GATE = 1.10
MODELS = {
    # name: (lane, dataset, trees, depth)
    "rf-higgs-100x16": ("rf", "higgs", 100, 16),
    "et-higgs-100x16": ("et", "higgs", 100, 16),
    "rf-covtype-100x16": ("rf", "covtype", 100, 16),
    "et-year-100x16": ("et", "year", 100, 16),
    "rf-higgs-500x16": ("rf", "higgs", 500, 16),
    # The two benchmark datasets of different kind (CONTRIBUTING.md
    # section 9): taxi is narrow (16 columns), Istella-S is wide (220). HIGGS
    # is retired as a result dataset and never decides; the regression
    # models have ONE output and so take the scalar grove kernel, which the
    # classifiers above never reach. Prediction rows are every row.
    "rf-taxi-100x16": ("rf", "taxi", 100, 16),
    "rf-taxireg-100x16": ("rf", "taxireg", 100, 16),
    "rf-istella-100x16": ("rf", "istella", 100, 16),
    "rf-istellareg-100x16": ("rf", "istellareg", 100, 16),
    # The ET regressors are kept as names but an ET REGRESSOR fit took 377.8 s
    # on taxireg (an RF classifier on the same rows takes 5 s; reported to
    # lane/forest-train-speed), so they are not in the default prepare list.
    "et-taxireg-100x16": ("et", "taxireg", 100, 16),
    "et-istellareg-100x16": ("et", "istellareg", 100, 16),
}


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _loaded_bindings():
    out = {}
    for name, mod in list(sys.modules.items()):
        if not name.startswith("mojolearn"):
            continue
        f = getattr(mod, "__file__", None)
        if f and f.endswith(".so") and os.path.exists(f):
            with open(f, "rb") as fh:
                out[os.path.basename(f)] = _sha(fh.read())[:16]
    return out


def _median(v):
    return statistics.median(v) if v else None


def prepare(args):
    import forest_speed_arm as forest
    os.makedirs(args.out, exist_ok=True)
    names = [n for n in args.models.split(",") if n] if args.models else list(MODELS)
    manifest = dict(models={}, higgs_train_rows=args.higgs_train)
    existing = os.path.join(args.out, "manifest.json")
    if os.path.exists(existing):
        # a later prepare adds models; it never drops the ones already saved
        with open(existing) as fh:
            manifest = json.load(fh)
    datasets = {}
    for name in names:
        lane, dataset, trees, depth = MODELS[name]
        if dataset not in datasets:
            rows_cap = args.higgs_train if dataset == "higgs" else None
            data = forest.spec.load_dataset(dataset, "shipped", rows_cap)
            forest.prepare_our_inputs(data)
            if dataset == "higgs":
                x = data._ours_Xtest
            else:
                x = np.ascontiguousarray(np.concatenate([data.X_train, data.X_test]), dtype=np.float32)
            xp = os.path.join(args.out, f"x_{dataset}.npy")
            np.save(xp, x)
            np.save(os.path.join(args.out, f"xtrain_{dataset}.npy"), data._ours_X)
            np.save(os.path.join(args.out, f"ytrain_{dataset}.npy"), data._ours_y)
            datasets[dataset] = (data, f"x_{dataset}.npy")
            print(f"prepare rows {dataset} predict {x.shape} sha256 {_sha(x.tobytes())[:16]} train {data._ours_X.shape} -> {xp}", flush=True)
        data, xname = datasets[dataset]
        cfg = forest.spec.lane_config(lane, "shipped")
        cfg["n_estimators"] = trees
        cfg["max_depth"] = depth
        arm = forest.OUR_BUILDERS[lane](lane, cfg, data)
        model = arm.make()
        t0 = time.perf_counter()
        arm.fit(model, data)
        arm.sync()
        fit_s = time.perf_counter() - t0
        model.inference_engine = "sequential"
        model.save(os.path.join(args.out, f"{name}.seq.npz"))
        model.inference_engine = "parallel_groves"
        path = os.path.join(args.out, f"{name}.npz")
        model.save(path)
        manifest["models"][name] = dict(
            path=path, lane=lane, dataset=dataset, trees=trees, depth=depth, x=xname,
            fit_s=round(fit_s, 1), task=data.task, n_classes=int(getattr(data, "n_classes", 0) or 0),
            nodes=int(model._colid.size), outputs=int(model._num_outputs),
            features=int(model.n_features_in_), config={k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in cfg.items()})
        print(f"prepare {name} fit {fit_s:.1f}s nodes {model._colid.size} outputs {model._num_outputs} -> {path}", flush=True)
        with open(os.path.join(args.out, "manifest.json"), "w") as fh:
            json.dump(manifest, fh, indent=1)


def _class_of(path):
    import mojolearn as ml
    arrays = ml._serialize.read_npz(path, (
        "mojolearn-randomforest-1-parallel-groves-1", "mojolearn-extratrees-1-parallel-groves-1",
        "mojolearn-randomforest-1", "mojolearn-extratrees-1"))
    return getattr(ml, ml._serialize.scalar_str(arrays, "estimator"))


def time_path(args):
    import mojolearn as ml
    x = np.ascontiguousarray(np.load(args.x), dtype=np.float32)
    if args.rows:
        x = x[:args.rows]
    if args.path.startswith("host"):
        model = ml.host_model(args.model)
        if args.path == "host-groves" and model.inference_engine != "parallel_groves":
            raise SystemExit(f"{args.model} is not a parallel_groves archive")
    else:
        model = _class_of(args.model).load(args.model)
        model.inference_engine = "parallel_groves" if args.path == "groves" else "sequential"
    classifier = hasattr(model, "classes_")
    method = model.predict_proba if classifier else model.predict

    def digest(r):
        r = np.asarray(r)
        return _sha(r.tobytes()) + f":{r.dtype}:{tuple(r.shape)}"

    t0 = time.perf_counter()
    warm = method(x)
    warm_ms = (time.perf_counter() - t0) * 1000.0
    hashes = [digest(warm)]
    if args.save_prediction and args.label == "after" and args.path == "groves":
        stem = os.path.join(args.save_prediction, os.path.basename(args.model)[:-4])
        if not os.path.exists(stem + ".groves.pred.npy"):
            np.save(stem + ".groves.pred.npy", np.asarray(warm))
    single = []
    for _ in range(args.rounds):
        t0 = time.perf_counter()
        r = method(x)
        single.append((time.perf_counter() - t0) * 1000.0)
        hashes.append(digest(r))
    blocks = []
    for _ in range(args.rounds):
        t0 = time.perf_counter()
        kept = [method(x) for _ in range(args.calls)]
        blocks.append((time.perf_counter() - t0) * 1000.0 / args.calls)
        hashes.extend(digest(r) for r in kept)
        del kept
    s_spread = max(single) / min(single)
    b_spread = max(blocks) / min(blocks)
    record = dict(
        path=args.path, model=os.path.abspath(args.model), x=os.path.abspath(args.x),
        rows=int(x.shape[0]), features=int(x.shape[1]), outputs=int(np.asarray(warm).reshape(len(warm), -1).shape[1]),
        engine=getattr(model, "inference_engine", None), warmup_ms=warm_ms,
        single_ms=single, single_median_ms=_median(single), single_spread=s_spread,
        single_stable=(s_spread <= SPREAD_GATE and len(single) >= 5),
        calls_per_block=args.calls, block_ms_per_call=blocks, block_median_ms=_median(blocks),
        block_spread=b_spread, block_stable=(b_spread <= SPREAD_GATE and len(blocks) >= 5),
        hash=hashes[0], hashes_equal=(len(set(hashes)) == 1),
        vendor=ml.vendor(), numeric_mode=ml.numeric_mode(),
        package_dir=os.path.dirname(ml.__file__), bindings=_loaded_bindings(),
        threads_env=os.environ.get("MOJOLEARN_CPU_THREADS", ""), label=args.label,
        cpu_count=os.cpu_count(),
    )
    with open(args.json, "w") as fh:
        json.dump(record, fh, indent=1)
    print(f"time {args.label} {args.path} {os.path.basename(args.model)} rows={x.shape[0]} "
          f"single={record['single_median_ms']:.2f}ms spread={s_spread:.3f} "
          f"block8={record['block_median_ms']:.2f}ms/call spread={b_spread:.3f} "
          f"stable={record['single_stable']}/{record['block_stable']} hash={hashes[0][:16]} same={record['hashes_equal']}")


def _profile_stages(log_path):
    """The mean of each FOREST_PROFILE stage over the lines after the first
    (the warmup), in ms."""
    lines = []
    with open(log_path) as fh:
        for line in fh:
            if line.startswith("FOREST_PROFILE"):
                toks = line.split()
                lines.append({toks[i]: int(toks[i + 1]) for i in range(1, len(toks) - 1, 2)
                              if toks[i].endswith("_ns")})
    if len(lines) < 2:
        return None
    lines = lines[1:]
    return {k[:-3] + "_ms": round(statistics.mean(l[k] for l in lines) / 1e6, 3) for k in lines[0]}


def summarize(args):
    recs = []
    for pat in args.json:
        for p in glob.glob(pat):
            with open(p) as fh:
                r = json.load(fh)
            r["_file"] = p
            recs.append(r)
    keys = sorted(set((os.path.basename(r["model"]), r["path"]) for r in recs))
    table = []
    for model, path in keys:
        row = dict(model=model, path=path)
        by_label = {}
        for r in recs:
            if (os.path.basename(r["model"]), r["path"]) == (model, path):
                by_label.setdefault(r["label"], []).append(r)
        hashes = set(r["hash"] for rs in by_label.values() for r in rs)
        row["hashes_equal_across_arms"] = len(hashes) == 1
        row["arms"] = {}
        for label, rs in by_label.items():
            row["arms"][label] = dict(
                processes=len(rs),
                single_median_ms=_median([r["single_median_ms"] for r in rs]),
                single_spreads=[round(r["single_spread"], 3) for r in rs],
                block_median_ms=_median([r["block_median_ms"] for r in rs]),
                block_spreads=[round(r["block_spread"], 3) for r in rs],
                stable_single=all(r["single_stable"] and r["hashes_equal"] for r in rs),
                stable_block=all(r["block_stable"] and r["hashes_equal"] for r in rs),
                rows=rs[0]["rows"], threads=rs[0].get("threads_env", ""),
                profile=_profile_stages(rs[0]["_file"][:-5] + ".log") if label == "v-profile" and os.path.exists(rs[0]["_file"][:-5] + ".log") else None,
            )
        if args.before in row["arms"]:
            b = row["arms"][args.before]
            for label, a in row["arms"].items():
                if label == args.before:
                    continue
                for scope in ("single", "block"):
                    ok = b[f"stable_{scope}"] and a[f"stable_{scope}"] and row["hashes_equal_across_arms"]
                    a[f"ratio_before_over_this_{scope}"] = (
                        round(b[f"{scope}_median_ms"] / a[f"{scope}_median_ms"], 3) if ok else None)
        table.append(row)
    print(json.dumps(table, indent=1))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(table, fh, indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--out", required=True)
    p.add_argument("--higgs-train", type=int, default=1000000)
    p.add_argument("--models", default="")
    t = sub.add_parser("time")
    t.add_argument("--model", required=True)
    t.add_argument("--x", required=True)
    t.add_argument("--path", required=True, choices=("groves", "sequential", "host-groves"))
    t.add_argument("--rounds", type=int, default=7)
    t.add_argument("--calls", type=int, default=8)
    t.add_argument("--rows", type=int, default=0)
    t.add_argument("--json", required=True)
    t.add_argument("--label", default="")
    t.add_argument("--save-prediction", default="")
    s = sub.add_parser("summarize")
    s.add_argument("json", nargs="+")
    s.add_argument("--before", default="before")
    s.add_argument("--out", default="")
    args = ap.parse_args()
    if args.cmd == "prepare":
        prepare(args)
    elif args.cmd == "time":
        time_path(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
