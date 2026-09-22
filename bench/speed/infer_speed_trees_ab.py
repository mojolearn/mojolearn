#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Inference timing for the forest and GBDT lanes, one path per process
(lane/infer-speed-trees, 2026-09-17).

Two modes. `prepare` trains the models the timing needs ONCE on the box's
GPU, saves each as its estimator's own archive and writes the fixed
prediction rows beside them, so every timed process reads the same bytes.
`time` loads one saved model in the checkout `PYTHONPATH` names, predicts
the fixed rows through ONE path, and writes the per-call milliseconds, the
median, the max/min spread, and the SHA-256 of the output bytes. A BEFORE
binary and an AFTER binary are therefore timed in ALTERNATING PROCESSES
over the same inputs, never in one process, because they are two builds
of the same module name.

    PYTHONPATH=python python3 bench/speed/infer_speed_trees_ab.py prepare \\
        --out /root/ab --rows 1000000 --gbdt-iterations 100,1000
    PYTHONPATH=python python3 bench/speed/infer_speed_trees_ab.py time \\
        --model /root/ab/rf-reg-100x16.npz --x /root/ab/x_taxireg.npy \\
        --path host-predict --rounds 5 --json /root/ab/after/rf-reg.host.json

Paths: `gpu-predict` and `gpu-proba` are the public estimator class's
`predict` / `predict_proba` (the GPU classes, whose forest default engine is
the sequential host walk); `host-predict` and `host-proba` are
`mojolearn.host_model(<file>)`, the CPU inference door. `gbdt-parse` times
the GBDT binding's model-text parse alone (`gbdt_model_dim`), a diagnostic
for how much of a GPU `predict` call is the parse.

The gate is at
least five timed calls after one warmup and a max/min spread of at most
1.10 per arm, output hashes equal across arms and rounds. A process whose
spread exceeds the gate is written with `stable: false`; the summary
refuses to quote a ratio from it. The loaded binaries are hashed FROM
INSIDE the process (the `.so` each module reports as its file), so a
result names the build that produced it.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tools"))

SPREAD_GATE = 1.10


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _file_sha(path):
    with open(path, "rb") as fh:
        return _sha(fh.read())


def _loaded_bindings():
    """The `.so` files this process has loaded under the mojolearn package,
    by digest, read from the modules themselves."""
    out = {}
    for name, mod in list(sys.modules.items()):
        if not name.startswith("mojolearn"):
            continue
        f = getattr(mod, "__file__", None)
        if f and f.endswith(".so") and os.path.exists(f):
            out[os.path.basename(f)] = _file_sha(f)[:16]
    return out


def prepare(args):
    import mojolearn as ml
    from speed_gbdt_arm import load_taxi
    os.makedirs(args.out, exist_ok=True)
    rows = args.rows
    cls = load_taxi("shipped", rows_cap=args.train_rows, regression=False)
    reg = load_taxi("shipped", rows_cap=args.train_rows, regression=True)
    # The fixed prediction rows: the LAST `rows` rows of each table in
    # temporal order (the shipped held-out block is 500k; a larger floor
    # takes the rows just before it). Timing does not need held-out rows.
    for name, data in (("taxi", cls), ("taxireg", reg)):
        full = np.concatenate([data.X_train, data.X_test]) if rows > data.X_test.shape[0] else data.X_test
        x = np.ascontiguousarray(full[-rows:], dtype=np.float32)
        path = os.path.join(args.out, f"x_{name}.npy")
        np.save(path, x)
        print(f"prepare rows {name} {x.shape} sha256 {_sha(x.tobytes())[:16]} -> {path}")
    manifest = dict(rows=rows, train_rows=args.train_rows, models={})
    t0 = time.perf_counter()
    m = ml.RandomForestRegressor(n_estimators=args.trees, max_depth=args.depth,
                                 numeric_mode="identical", random_state=7)
    m.fit(reg.X_train, reg.y_train)
    p = os.path.join(args.out, f"rf-reg-{args.trees}x{args.depth}.npz")
    m.save(p)
    manifest["models"]["rf-reg"] = dict(path=p, kind="rf-reg", fit_s=time.perf_counter() - t0, x="x_taxireg.npy")
    print(f"prepare rf-reg fit {time.perf_counter() - t0:.1f}s -> {p}")
    t0 = time.perf_counter()
    m = ml.ExtraTreesRegressor(n_estimators=args.trees, max_depth=args.depth,
                               numeric_mode="identical", random_state=7)
    m.fit(reg.X_train, reg.y_train)
    p = os.path.join(args.out, f"et-reg-{args.trees}x{args.depth}.npz")
    m.save(p)
    manifest["models"]["et-reg"] = dict(path=p, kind="et-reg", fit_s=time.perf_counter() - t0, x="x_taxireg.npy")
    print(f"prepare et-reg fit {time.perf_counter() - t0:.1f}s -> {p}")
    for iters in [int(v) for v in args.gbdt_iterations.split(",") if v]:
        t0 = time.perf_counter()
        m = ml.GradientBoosting(loss="Logloss", n_estimators=iters, max_depth=6, numeric_mode="identical")
        m.fit(cls.X_train, cls.y_train)
        p = os.path.join(args.out, f"gbdt-logloss-{iters}.npz")
        m.save(p)
        manifest["models"][f"gbdt-logloss-{iters}"] = dict(path=p, kind="gbdt", fit_s=time.perf_counter() - t0, x="x_taxi.npy")
        print(f"prepare gbdt-logloss-{iters} fit {time.perf_counter() - t0:.1f}s -> {p}")
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)


def _load_gpu_model(kind, path):
    import mojolearn as ml
    if kind == "rf-reg":
        return ml.RandomForestRegressor.load(path)
    if kind == "et-reg":
        return ml.ExtraTreesRegressor.load(path)
    if kind == "gbdt":
        return ml.GradientBoosting.load(path)
    raise SystemExit(f"unknown kind {kind}")


def time_path(args):
    import mojolearn as ml
    x = np.load(args.x)
    x = np.ascontiguousarray(x, dtype=np.float32)
    if args.path.startswith("host"):
        model = ml.host_model(args.model)
    else:
        model = _load_gpu_model(args.kind, args.model)
    if args.path == "gbdt-parse":
        binding = model._bind("_mojolearn_gbdt")
        text = model.model_
        def call():
            return int(binding.gbdt_model_dim(text))
    elif args.path.endswith("proba"):
        def call():
            return model.predict_proba(x)
    else:
        def call():
            return model.predict(x)

    def digest(result):
        if isinstance(result, int):
            return f"dim:{result}"
        return _sha(result.tobytes()) + f":{result.dtype}:{tuple(result.shape)}"

    t0 = time.perf_counter()
    warm = call()
    warm_ms = (time.perf_counter() - t0) * 1000.0
    hashes = [digest(warm)]
    rounds = []
    for _ in range(args.rounds):
        t0 = time.perf_counter()
        r = call()
        rounds.append((time.perf_counter() - t0) * 1000.0)
        hashes.append(digest(r))
    ordered = sorted(rounds)
    median = ordered[len(ordered) // 2] if len(ordered) % 2 else 0.5 * (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2])
    spread = ordered[-1] / ordered[0] if ordered[0] > 0 else float("inf")
    record = dict(
        path=args.path, kind=args.kind, model=os.path.abspath(args.model), x=os.path.abspath(args.x),
        rows=int(x.shape[0]), features=int(x.shape[1]), rounds=rounds, warmup_ms=warm_ms,
        median_ms=median, spread=spread, stable=(spread <= SPREAD_GATE and len(rounds) >= 5),
        hash=hashes[0], hashes_equal=(len(set(hashes)) == 1),
        vendor=ml.vendor(), numeric_mode=ml.numeric_mode(),
        package_dir=os.path.dirname(ml.__file__), bindings=_loaded_bindings(),
        threads_env=os.environ.get("MOJOLEARN_CPU_THREADS", ""), label=args.label,
        cpu_count=os.cpu_count(),
    )
    with open(args.json, "w") as fh:
        json.dump(record, fh, indent=1)
    print(f"time {args.label} {args.path} {args.kind} rows={x.shape[0]} median={median:.2f}ms "
          f"spread={spread:.3f} stable={record['stable']} hash={hashes[0][:16]} same={record['hashes_equal']}")


def summarize(args):
    """Pair BEFORE and AFTER JSONs of the same path and kind into a table:
    the median of each side's process medians, and the ratio, only where
    every process is stable and every hash agrees."""
    import glob
    recs = []
    for pat in args.json:
        for p in glob.glob(pat):
            with open(p) as fh:
                r = json.load(fh)
                r["_file"] = p
                recs.append(r)
    keys = sorted(set((r["kind"], r["path"], os.path.basename(r["model"])) for r in recs))
    table = []
    for kind, path, model in keys:
        row = dict(kind=kind, path=path, model=model)
        by_label = {}
        for r in recs:
            if (r["kind"], r["path"], os.path.basename(r["model"])) == (kind, path, model):
                by_label.setdefault(r["label"], []).append(r)
        hashes = set(r["hash"] for rs in by_label.values() for r in rs)
        row["hashes_equal_across_arms"] = len(hashes) == 1
        for label, rs in by_label.items():
            meds = sorted(r["median_ms"] for r in rs)
            row[label] = dict(processes=len(rs), median_ms=meds[len(meds) // 2],
                              spreads=[round(r["spread"], 3) for r in rs],
                              stable=all(r["stable"] and r["hashes_equal"] for r in rs),
                              rows=rs[0]["rows"], threads=rs[0].get("threads_env", ""))
        if args.before in row and args.after in row:
            b, a = row[args.before], row[args.after]
            qualified = b["stable"] and a["stable"] and row["hashes_equal_across_arms"]
            row["ratio_before_over_after"] = (b["median_ms"] / a["median_ms"]) if qualified else None
            row["qualified"] = qualified
        table.append(row)
    print(json.dumps(table, indent=1))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(table, fh, indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--out", required=True)
    p.add_argument("--rows", type=int, default=1000000)
    p.add_argument("--train-rows", type=int, default=1000000)
    p.add_argument("--trees", type=int, default=100)
    p.add_argument("--depth", type=int, default=16)
    p.add_argument("--gbdt-iterations", default="100,1000")
    t = sub.add_parser("time")
    t.add_argument("--model", required=True)
    t.add_argument("--x", required=True)
    t.add_argument("--kind", required=True, choices=("rf-reg", "et-reg", "gbdt"))
    t.add_argument("--path", required=True,
                   choices=("gpu-predict", "gpu-proba", "host-predict", "host-proba", "gbdt-parse"))
    t.add_argument("--rounds", type=int, default=5)
    t.add_argument("--json", required=True)
    t.add_argument("--label", default="")
    s = sub.add_parser("summarize")
    s.add_argument("json", nargs="+")
    s.add_argument("--before", default="before")
    s.add_argument("--after", default="after")
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
