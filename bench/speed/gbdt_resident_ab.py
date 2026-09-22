#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Speed evidence for the device-resident GBDT model (lane/gbdt-resident-
predict, 2026-09-17, DEVIATION 2980), interleaved arms in ONE process.

`prepare` trains the models the timing needs once on the box's GPU, saves
each as its estimator's archive, writes the fixed prediction rows and
their targets beside them, and (with `--catboost`) trains a CatBoost model
of the same shape on the same rows and saves it as `.cbm`. `time` loads one
saved mojolearn model, predicts the fixed rows through `predict` or
`predict_proba`, and INTERLEAVES the two arms round by round: `resident`
(the door DEVIATION 2980 adds, `mojolearn.ensemble.GBDT_RESIDENT` True) and
`percall` (the per-call parse, the same binary with the switch off). Both
arms therefore run the same binary, the same bytes and the same process;
what differs is only whether the parsed model stays on the device. Each
round times `--calls` consecutive calls and records the per-call
milliseconds, the SHA-256 of the output bytes, and the quality of the
output against the saved target (RMSE for a regression model, log loss for
Logloss, accuracy for MultiClass). `catboost` times CatBoost's own
`predict` over the same rows with `task_type="GPU"` and `"CPU"`, the same
gate. `summarize` pairs the JSONs into one table.

    PYTHONPATH=python python3 bench/speed/gbdt_resident_ab.py prepare \\
        --out /root/ab --rows 1000000 --datasets taxi,taxireg,higgs,covtype --catboost
    PYTHONPATH=python python3 bench/speed/gbdt_resident_ab.py time \\
        --model /root/ab/taxi-logloss-1000.npz --x /root/ab/x_taxi.npy --y /root/ab/y_taxi.npy \\
        --path proba --rounds 5 --calls 1 --json /root/ab/out/taxi-1000.proba.1.json
    PYTHONPATH=python python3 bench/speed/gbdt_resident_ab.py catboost \\
        --model /root/ab/catboost-taxi-1000.cbm --x /root/ab/x_taxi.npy --path predict \\
        --devices gpu,cpu --json /root/ab/out/catboost-taxi-1000.predict.json

The gate is at least five
timed rounds after one warmup per arm, max/min spread at most 1.10 per
arm, output hashes equal across arms and rounds. A process whose spread
exceeds the gate is written `stable: false` and the summary refuses to
quote a ratio from it. The loaded binaries are hashed from inside the
process, and the resident arm records the registry's view of the handle
(`gbdt_resident_info`), so a result names the build and the door that
produced it.
"""
import argparse
import glob
import hashlib
import json
import math
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
    out = {}
    for name, mod in list(sys.modules.items()):
        if not name.startswith("mojolearn"):
            continue
        f = getattr(mod, "__file__", None)
        if f and f.endswith(".so") and os.path.exists(f):
            out[os.path.basename(f)] = _file_sha(f)[:16]
    return out


def _data_root():
    return os.environ.get("GBM_BENCH_DATA", os.path.join(os.path.expanduser("~"), "gbm-bench-data"))


# ---------------------------------------------------------------- datasets
# Every table comes from the R2 store's decoded npz (tools/dataset_store.sh,
# DEVIATION 2704); nothing here downloads. taxi goes through
# tools/speed_gbdt_arm.py's `load_taxi` so the classification label (a tip of
# 20 percent or more on a card-paid trip) and the fare target are the board's.

def _tail_rows(x, y, rows):
    rows = min(rows, x.shape[0])
    return np.ascontiguousarray(x[-rows:], dtype=np.float32), np.ascontiguousarray(y[-rows:], dtype=np.float32)


def load_dataset(name, train_rows, rows):
    """`(x_train, y_train, x_pred, y_pred, task, n_classes)`; the prediction
    rows are the LAST `rows` rows of the table (for taxi, of train + test in
    temporal order, as bench/speed/infer_speed_trees_ab.py takes them)."""
    if name in ("taxi", "taxireg"):
        from speed_gbdt_arm import load_taxi
        d = load_taxi("shipped", rows_cap=train_rows, regression=(name == "taxireg"))
        full_x = np.concatenate([d.X_train, d.X_test]) if rows > d.X_test.shape[0] else d.X_test
        full_y = np.concatenate([d.y_train, d.y_test]) if rows > d.X_test.shape[0] else d.y_test
        xp, yp = _tail_rows(full_x, full_y, rows)
        task = "regression" if name == "taxireg" else "binary"
        return d.X_train, d.y_train, xp, yp, task, 0
    if name == "higgs":
        z = np.load(os.path.join(_data_root(), "higgs", "higgs_speed.npz"))
        x, y = z["x"], z["y"]
        xt = np.ascontiguousarray(x[:train_rows], dtype=np.float32)
        yt = np.ascontiguousarray(y[:train_rows], dtype=np.float32)
        xp, yp = _tail_rows(x, y, rows)
        return xt, yt, xp, yp, "binary", 0
    if name == "covtype":
        z = np.load(os.path.join(_data_root(), "covtype", "covtype_speed.npz"))
        x = np.ascontiguousarray(z["x"], dtype=np.float32)
        y = (z["target"].astype(np.int64) - 1).astype(np.float32)
        n_train = min(train_rows, x.shape[0])
        xp, yp = _tail_rows(x, y, rows)
        return x[:n_train], y[:n_train], xp, yp, "multiclass", 7
    raise SystemExit(f"unknown dataset {name}")


def prepare(args):
    import mojolearn as ml
    os.makedirs(args.out, exist_ok=True)
    iters_list = [int(v) for v in args.gbdt_iterations.split(",") if v]
    manifest = dict(rows=args.rows, train_rows=args.train_rows, models={}, datasets={})
    for name in [d for d in args.datasets.split(",") if d]:
        xt, yt, xp, yp, task, n_classes = load_dataset(name, args.train_rows, args.rows)
        xpath = os.path.join(args.out, f"x_{name}.npy")
        ypath = os.path.join(args.out, f"y_{name}.npy")
        np.save(xpath, xp)
        np.save(ypath, yp)
        manifest["datasets"][name] = dict(x=xpath, y=ypath, rows=int(xp.shape[0]), features=int(xp.shape[1]),
                                          train_rows=int(xt.shape[0]), task=task, n_classes=n_classes,
                                          x_sha256=_sha(xp.tobytes())[:16])
        print(f"prepare {name}: train {xt.shape} predict {xp.shape} task {task} x sha {_sha(xp.tobytes())[:16]}")
        loss = {"regression": "RMSE", "binary": "Logloss", "multiclass": "MultiClass"}[task]
        for iters in iters_list:
            if task == "multiclass" and iters > args.multiclass_max_iterations:
                continue
            t0 = time.perf_counter()
            m = ml.GradientBoosting(loss=loss, n_estimators=iters, max_depth=6, numeric_mode="identical")
            m.fit(xt, yt)
            p = os.path.join(args.out, f"{name}-{loss.lower()}-{iters}.npz")
            m.save(p)
            fit_s = time.perf_counter() - t0
            manifest["models"][f"{name}-{loss.lower()}-{iters}"] = dict(
                path=p, dataset=name, loss=loss, iterations=iters, depth=6, fit_s=fit_s,
                x=xpath, y=ypath, model_sha256=_file_sha(p)[:16])
            print(f"prepare {name} {loss} {iters}: fit {fit_s:.1f}s -> {p}")
            if args.catboost:
                import catboost
                params = dict(iterations=iters, depth=6, learning_rate=0.03, l2_leaf_reg=3.0,
                              border_count=128, random_seed=0, boosting_type="Plain",
                              bootstrap_type="No", task_type="GPU", devices="0",
                              verbose=False, allow_writing_files=False)
                t0 = time.perf_counter()
                if task == "regression":
                    cb = catboost.CatBoostRegressor(loss_function="RMSE", **params)
                elif task == "binary":
                    cb = catboost.CatBoostClassifier(loss_function="Logloss", **params)
                else:
                    cb = catboost.CatBoostClassifier(loss_function="MultiClass", **params)
                cb.fit(xt, yt)
                cp = os.path.join(args.out, f"catboost-{name}-{loss.lower()}-{iters}.cbm")
                cb.save_model(cp)
                cb_s = time.perf_counter() - t0
                manifest["models"][f"{name}-{loss.lower()}-{iters}"]["catboost"] = dict(
                    path=cp, fit_s=cb_s, version=catboost.__version__, params=params,
                    tree_count=int(cb.tree_count_))
                print(f"prepare catboost {name} {loss} {iters}: fit {cb_s:.1f}s trees {cb.tree_count_} -> {cp}")
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)


# ---------------------------------------------------------------- quality

def _quality(task, path, out, y):
    """One number per output, against the saved target, so a speed cell
    also says the model predicts something."""
    if y is None:
        return None
    out = np.asarray(out)
    if task == "regression":
        return dict(rmse=float(math.sqrt(float(np.mean((out.astype(np.float64) - y) ** 2)))))
    if task == "binary":
        if path == "proba":
            p = np.clip(out[:, 1].astype(np.float64), 1e-15, 1 - 1e-15)
        else:
            p = 1.0 / (1.0 + np.exp(-out.astype(np.float64)))
            p = np.clip(p, 1e-15, 1 - 1e-15)
        ll = -float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        return dict(logloss=ll, accuracy=float(np.mean((p >= 0.5) == (y >= 0.5))))
    if task == "multiclass":
        if path == "proba":
            labels = np.argmax(out, axis=1)
        else:
            # raw approxes are `n_classes - 1` wide with the last class pinned at zero
            full = np.concatenate([out, np.zeros((out.shape[0], 1), dtype=out.dtype)], axis=1)
            labels = np.argmax(full, axis=1)
        return dict(accuracy=float(np.mean(labels == y.astype(np.int64))))
    return None


def _stats(rounds):
    ordered = sorted(rounds)
    n = len(ordered)
    median = ordered[n // 2] if n % 2 else 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])
    spread = ordered[-1] / ordered[0] if ordered[0] > 0 else float("inf")
    return median, spread


# ---------------------------------------------------------------- time

def time_path(args):
    import mojolearn as ml
    from mojolearn import ensemble
    x = np.load(args.x)
    x = np.asfortranarray(x, dtype=np.float32) if args.order == "F" else np.ascontiguousarray(x, dtype=np.float32)
    y = np.load(args.y) if args.y else None
    model = ml.GradientBoosting.load(args.model)
    task = {"RMSE": "regression", "Logloss": "binary", "CrossEntropy": "binary"}.get(model.loss, "multiclass")
    binding = model._bind("_mojolearn_gbdt")
    has_door = hasattr(binding, "gbdt_resident_prepare")
    arms = [a for a in args.arms.split(",") if a]
    if "resident" in arms and not has_door:
        raise SystemExit("REFUSING: the loaded binding has no gbdt_resident_prepare; this build cannot time the resident arm")

    def call():
        if args.path == "proba":
            return model.predict_proba(x)
        return model.predict(x)

    def digest(r):
        return _sha(r.tobytes()) + f":{r.dtype}:{tuple(r.shape)}"

    def set_arm(arm):
        ensemble.GBDT_RESIDENT = (arm == "resident")

    # one warmup per arm, in order; the resident warmup is where the parse
    # and the upload happen, and it is recorded on its own
    warm = {}
    hashes = {}
    quality = {}
    for arm in arms:
        set_arm(arm)
        t0 = time.perf_counter()
        r = call()
        warm[arm] = (time.perf_counter() - t0) * 1000.0
        hashes[arm] = [digest(r)]
        quality[arm] = _quality(task, args.path, r, y)
        if arm == "resident":
            if getattr(model, "_resident", None) is None:
                raise SystemExit("REFUSING: the resident arm did not prepare a handle")
        elif getattr(model, "_resident", None) is not None and arms.index(arm) == 0:
            raise SystemExit("REFUSING: the percall arm prepared a handle")
    rounds = {arm: [] for arm in arms}
    for _ in range(args.rounds):
        for arm in arms:
            set_arm(arm)
            t0 = time.perf_counter()
            for _c in range(args.calls):
                r = call()
            rounds[arm].append((time.perf_counter() - t0) * 1000.0 / args.calls)
            hashes[arm].append(digest(r))
    info = None
    if "resident" in arms and getattr(model, "_resident", None) is not None:
        info = [int(v) for v in binding.gbdt_resident_info(model._resident[3])]
    record = dict(
        path=args.path, model=os.path.abspath(args.model), x=os.path.abspath(args.x), order=args.order,
        rows=int(x.shape[0]), features=int(x.shape[1]), loss=model.loss, task=task, calls=args.calls,
        n_trees=None if info is None else info[4], approx_dim=int(model.approx_dim_),
        arms={}, label=args.label, vendor=ml.vendor(), numeric_mode=ml.numeric_mode(),
        package_dir=os.path.dirname(ml.__file__), bindings=_loaded_bindings(),
        threads_env=os.environ.get("MOJOLEARN_CPU_THREADS", ""), cpu_count=os.cpu_count(),
        resident_info=info,
    )
    all_hashes = set()
    for arm in arms:
        median, spread = _stats(rounds[arm])
        all_hashes.update(hashes[arm])
        record["arms"][arm] = dict(rounds=rounds[arm], warmup_ms=warm[arm], median_ms=median, spread=spread,
                                   stable=(spread <= SPREAD_GATE and len(rounds[arm]) >= 5),
                                   hash=hashes[arm][0], hashes_equal=(len(set(hashes[arm])) == 1),
                                   quality=quality[arm])
    record["hashes_equal_across_arms"] = (len(all_hashes) == 1)
    if "resident" in record["arms"] and "percall" in record["arms"]:
        a, b = record["arms"]["resident"], record["arms"]["percall"]
        q = a["stable"] and b["stable"] and record["hashes_equal_across_arms"]
        record["ratio_percall_over_resident"] = (b["median_ms"] / a["median_ms"]) if q else None
        record["qualified"] = q
    with open(args.json, "w") as fh:
        json.dump(record, fh, indent=1)
    parts = [f"{arm} median={record['arms'][arm]['median_ms']:.2f}ms spread={record['arms'][arm]['spread']:.3f}"
             for arm in arms]
    print(f"time {args.label} {args.path} {os.path.basename(args.model)} rows={x.shape[0]} calls={args.calls} "
          f"{' '.join(parts)} same={record['hashes_equal_across_arms']} ratio={record.get('ratio_percall_over_resident')}")


# ---------------------------------------------------------------- catboost

def time_catboost(args):
    import catboost
    x = np.load(args.x)
    x = np.ascontiguousarray(x, dtype=np.float32)
    y = np.load(args.y) if args.y else None
    with open(os.path.join(os.path.dirname(args.model), "manifest.json")) as fh:
        manifest = json.load(fh)
    entry = None
    for m in manifest["models"].values():
        if m.get("catboost", {}).get("path") == os.path.abspath(args.model) or \
                os.path.basename(m.get("catboost", {}).get("path", "")) == os.path.basename(args.model):
            entry = m
    if entry is None:
        raise SystemExit(f"{args.model} is not in the manifest")
    loss = entry["loss"]
    task = {"RMSE": "regression", "Logloss": "binary"}.get(loss, "multiclass")
    if task == "regression":
        cb = catboost.CatBoostRegressor()
    else:
        cb = catboost.CatBoostClassifier()
    cb.load_model(args.model)
    prediction_type = "Probability" if args.path == "proba" else "RawFormulaVal"
    record = dict(path=args.path, model=os.path.abspath(args.model), x=os.path.abspath(args.x), rows=int(x.shape[0]),
                  features=int(x.shape[1]), loss=loss, task=task, calls=args.calls, n_trees=int(cb.tree_count_),
                  catboost=catboost.__version__, prediction_type=prediction_type, arms={}, label=args.label,
                  cpu_count=os.cpu_count())
    for dev in [d for d in args.devices.split(",") if d]:
        task_type = "GPU" if dev.startswith("gpu") else "CPU"
        threads = 1 if dev.endswith("1") else -1

        def call():
            return cb.predict(x, prediction_type=prediction_type, task_type=task_type, thread_count=threads)

        try:
            t0 = time.perf_counter()
            r = call()
            warm = (time.perf_counter() - t0) * 1000.0
        except Exception as exc:  # noqa: BLE001
            record["arms"][dev] = dict(refused=f"{type(exc).__name__}: {exc}")
            print(f"catboost {dev}: refused {type(exc).__name__}: {exc}")
            continue
        hashes = [_sha(np.ascontiguousarray(r).tobytes())]
        rounds = []
        for _ in range(args.rounds):
            t0 = time.perf_counter()
            for _c in range(args.calls):
                r = call()
            rounds.append((time.perf_counter() - t0) * 1000.0 / args.calls)
            hashes.append(_sha(np.ascontiguousarray(r).tobytes()))
        median, spread = _stats(rounds)
        record["arms"][dev] = dict(rounds=rounds, warmup_ms=warm, median_ms=median, spread=spread,
                                   stable=(spread <= SPREAD_GATE and len(rounds) >= 5), hash=hashes[0],
                                   hashes_equal=(len(set(hashes)) == 1), task_type=task_type, thread_count=threads,
                                   quality=_quality(task, args.path, np.asarray(r), y),
                                   output_dtype=str(np.asarray(r).dtype), output_shape=list(np.asarray(r).shape))
        print(f"catboost {args.label} {dev} {args.path} {os.path.basename(args.model)} rows={x.shape[0]} "
              f"median={median:.2f}ms spread={spread:.3f} same={len(set(hashes)) == 1}")
    with open(args.json, "w") as fh:
        json.dump(record, fh, indent=1)


# ---------------------------------------------------------------- summarize

def summarize(args):
    recs = []
    for pat in args.json:
        for p in glob.glob(pat):
            with open(p) as fh:
                r = json.load(fh)
            if not isinstance(r, dict) or "arms" not in r:
                continue  # the manifest, a summary
            r["_file"] = p
            recs.append(r)
    table = []
    keys = sorted(set((os.path.basename(r["model"]), r["path"], r.get("calls", 1), r.get("order", ""), r.get("label", "")) for r in recs))
    for model, path, calls, order, label in keys:
        rs = [r for r in recs if (os.path.basename(r["model"]), r["path"], r.get("calls", 1), r.get("order", ""), r.get("label", "")) == (model, path, calls, order, label)]
        row = dict(model=model, path=path, calls=calls, order=order, label=label, rows=rs[0]["rows"], processes=len(rs),
                   n_trees=rs[0].get("n_trees"), loss=rs[0].get("loss"))
        arms = {}
        for r in rs:
            for arm, a in r["arms"].items():
                arms.setdefault(arm, []).append(a)
        hashes_all = set()
        for arm, lst in arms.items():
            if any("refused" in a for a in lst):
                row[arm] = dict(refused=[a.get("refused") for a in lst if "refused" in a])
                continue
            meds = sorted(a["median_ms"] for a in lst)
            row[arm] = dict(median_ms=meds[len(meds) // 2], spreads=[round(a["spread"], 3) for a in lst],
                            stable=all(a["stable"] and a["hashes_equal"] for a in lst),
                            quality=lst[0].get("quality"), hash=lst[0]["hash"][:16])
            hashes_all.update(a["hash"] for a in lst)
        row["mojolearn_hashes_equal"] = all(r.get("hashes_equal_across_arms", True) for r in rs if "resident" in r["arms"])
        if "resident" in row and "percall" in row and "refused" not in row["resident"]:
            q = row["resident"]["stable"] and row["percall"]["stable"] and row["mojolearn_hashes_equal"]
            row["ratio_percall_over_resident"] = (row["percall"]["median_ms"] / row["resident"]["median_ms"]) if q else None
            row["qualified"] = q
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
    p.add_argument("--gbdt-iterations", default="100,1000")
    p.add_argument("--multiclass-max-iterations", type=int, default=1000)
    p.add_argument("--datasets", default="taxi,taxireg")
    p.add_argument("--catboost", action="store_true")
    t = sub.add_parser("time")
    t.add_argument("--model", required=True)
    t.add_argument("--x", required=True)
    t.add_argument("--y", default="")
    t.add_argument("--path", required=True, choices=("predict", "proba"))
    t.add_argument("--arms", default="resident,percall")
    t.add_argument("--rounds", type=int, default=5)
    t.add_argument("--calls", type=int, default=1)
    t.add_argument("--order", default="C", choices=("C", "F"))
    t.add_argument("--json", required=True)
    t.add_argument("--label", default="")
    c = sub.add_parser("catboost")
    c.add_argument("--model", required=True)
    c.add_argument("--x", required=True)
    c.add_argument("--y", default="")
    c.add_argument("--path", required=True, choices=("predict", "proba"))
    c.add_argument("--devices", default="gpu,cpu,cpu1")
    c.add_argument("--rounds", type=int, default=5)
    c.add_argument("--calls", type=int, default=1)
    c.add_argument("--json", required=True)
    c.add_argument("--label", default="")
    s = sub.add_parser("summarize")
    s.add_argument("json", nargs="+")
    s.add_argument("--out", default="")
    args = ap.parse_args()
    if args.cmd == "prepare":
        prepare(args)
    elif args.cmd == "time":
        time_path(args)
    elif args.cmd == "catboost":
        time_catboost(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
