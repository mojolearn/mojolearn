#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The `parallel_groves` engine, GPU against host, bit for bit
(lane/forest-groves-cpu-and-speed, 2026-09-17).

tools/identity_break.py records every rf and et lane under the DEFAULT
engine, `sequential`. This script is the groves column: for every rf and et
lane and every fixture it fits the lane's own estimator (the lane function,
unchanged), switches the fitted forest to `parallel_groves`, records the GPU
groves predictions on the held-out rows, saves the model (a
`-parallel-groves-1` archive), loads that file through `mojolearn.host_model`
(the shipped forest host binding, CPU only) and compares the host groves
predictions with the GPU's, part by part. It also records the `sequential`
predictions of the same fit, so a cell says whether the two engines differ
on that fixture at all (a cell where they do not cannot tell a host engine
that silently ran the sequential walk from one that ran the groves fold).

    lanes   every rf and et lane in identity_break.LANES, --fixtures, one fit each
    large   the saved HIGGS and Covtype sized groves archives under --models-dir
            (written by tools/forest_groves_speed.py prepare) against their
            fixed prediction rows

Cells read IDENTICAL, DIVERGENT or REFUSED (a stage raised; the text is
kept). The host binary's sabotage read-backs are recorded, so a column from
a `MOJOLEARN_FOREST_HOST_SABOTAGE` or `MOJOLEARN_FOREST_GROVES_SABOTAGE`
build names itself; under either every cell is expected DIVERGENT, and a
cell the association arm cannot move is reported with its
`groves_vs_sequential` count so the fixture's blindness is on record.
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "python"))

FOREST_LANE_PREFIXES = ("rf-", "et-")


def _h(*arrays):
    m = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(np.asarray(a))
        if a.dtype == object or a.dtype.kind in "OUSV":
            raise TypeError(f"refusing dtype={a.dtype}")
        m.update(str(a.dtype).encode())
        m.update(str(a.shape).encode())
        m.update(a.tobytes())
    return m.hexdigest()[:16]


def _host_witness():
    """The forest host binary this process loads and its sabotage flags."""
    from mojolearn import _forest_host
    mod = _forest_host._load()
    path = getattr(mod, "__file__", _forest_host.binary_path())
    with open(path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()[:16]
    return dict(
        path=os.path.realpath(path), sha256=digest,
        forest_host_sabotage=bool(mod.forest_host_sabotage()),
        forest_host_groves_sabotage=bool(getattr(mod, "forest_host_groves_sabotage", lambda: False)()),
        has_groves_entries=callable(getattr(mod, "forest_host_groves_prepare", None)),
    )


def _parts_of(est, Xh):
    """The prediction parts a lane's estimator answers on held-out rows:
    `predict` always, `predict_proba` for a classifier."""
    parts = {"predict": np.asarray(est.predict(Xh))}
    if hasattr(est, "predict_proba") and hasattr(est, "classes_"):
        parts["proba"] = np.asarray(est.predict_proba(Xh))
    return parts


def _compare(gpu, host):
    verdict = {}
    for name in gpu:
        if name not in host:
            verdict[name] = "REFUSED: host answered no " + name
            continue
        a, b = gpu[name], host[name]
        if a.dtype != b.dtype or a.shape != b.shape:
            verdict[name] = f"DIVERGENT: dtype/shape {a.dtype}{a.shape} vs {b.dtype}{b.shape}"
        elif a.tobytes() == b.tobytes():
            verdict[name] = "IDENTICAL"
        else:
            differing = int(np.count_nonzero(a.view(np.uint8).reshape(a.shape[0], -1) != b.view(np.uint8).reshape(b.shape[0], -1)) > 0) if a.ndim else 0
            rows = int(np.count_nonzero((a.reshape(a.shape[0], -1) != b.reshape(b.shape[0], -1)).any(axis=1)))
            verdict[name] = f"DIVERGENT: {rows} of {a.shape[0]} rows differ"
    return verdict


def _cell(est, Xh, tmp, name):
    """One fit: GPU groves parts, sequential parts, host groves parts."""
    import mojolearn
    record = dict(error=None)
    engine_before = getattr(est, "inference_engine", "sequential")
    try:
        est.inference_engine = "sequential"
        seq = _parts_of(est, Xh)
        est.inference_engine = "parallel_groves"
        gpu = _parts_of(est, Xh)
        record["gpu_groves"] = {k: _h(v) for k, v in gpu.items()}
        record["sequential"] = {k: _h(v) for k, v in seq.items()}
        record["groves_vs_sequential_rows_differ"] = {
            k: int(np.count_nonzero((np.asarray(gpu[k]).reshape(len(gpu[k]), -1) != np.asarray(seq[k]).reshape(len(seq[k]), -1)).any(axis=1)))
            for k in gpu}
        path = os.path.join(tmp, f"{name}.npz")
        est.save(path)
        with open(path, "rb") as fh:
            record["model_sha256"] = hashlib.sha256(fh.read()).hexdigest()[:16]
        record["archive_format"] = str(mojolearn._serialize.scalar_str(
            mojolearn._serialize.read_npz(path, (
                "mojolearn-randomforest-1-parallel-groves-1", "mojolearn-extratrees-1-parallel-groves-1",
                "mojolearn-randomforest-1", "mojolearn-extratrees-1")), "format"))
    except Exception as exc:
        record["error"] = f"gpu: {type(exc).__name__}: {exc}"
        est.inference_engine = engine_before
        return record
    try:
        host = mojolearn.host_model(path)
        record["host_engine"] = host.inference_engine
        hp = _parts_of(host, Xh)
        record["host_groves"] = {k: _h(v) for k, v in hp.items()}
        record["verdict"] = _compare(gpu, hp)
        # a second load must answer the same bytes
        hp2 = _parts_of(mojolearn.host_model(path), Xh)
        record["host_reload_same"] = all(hp[k].tobytes() == hp2[k].tobytes() for k in hp)
    except Exception as exc:
        record["error"] = f"host: {type(exc).__name__}: {exc}"
        record["verdict"] = {k: f"REFUSED: {type(exc).__name__}: {exc}" for k in gpu}
    finally:
        est.inference_engine = engine_before
    return record


def run_lanes(args):
    import identity_break as ib
    import mojolearn as ml
    names = [n for n in ib.LANES if n.startswith(FOREST_LANE_PREFIXES)]
    if args.lanes:
        names = [n for n in args.lanes.split(",") if n]
    fixtures = [f for f in args.fixtures.split(",") if f]
    witness = _host_witness()
    out = dict(kind="lanes", vendor=ml.vendor(), numeric_mode=ml.numeric_mode(), host=witness,
               lanes=names, fixtures=fixtures, cells={}, threads=os.environ.get("MOJOLEARN_CPU_THREADS", ""))
    counts = dict(IDENTICAL=0, DIVERGENT=0, REFUSED=0)
    with tempfile.TemporaryDirectory(prefix="forest_groves_identity_") as tmp:
        for name in names:
            for kind in fixtures:
                X, yc, yr = ib.fixture(kind)
                Xh = ib.heldout(kind)
                key = f"{name}/{kind}"
                t0 = time.perf_counter()
                try:
                    fit = ib.LANES[name](ml, X, yc, yr, Xh)
                    est = fit.est
                    if est is None or not hasattr(est, "save"):
                        rec = dict(error=None, na="the lane returns no estimator (a scoring lane)", verdict={})
                        print(f"{key}: n/a, {rec['na']}", flush=True)
                        out["cells"][key] = rec
                        continue
                    rec = _cell(est, Xh, tmp, f"{name}-{kind}")
                except Exception as exc:
                    rec = dict(error=f"fit: {type(exc).__name__}: {exc}", verdict={"predict": f"REFUSED: fit: {type(exc).__name__}: {exc}"})
                rec["seconds"] = round(time.perf_counter() - t0, 2)
                out["cells"][key] = rec
                for part, v in rec.get("verdict", {}).items():
                    counts[v.split(":")[0]] += 1
                    print(f"{key} {part}: {v}"
                          + (f" (groves vs sequential rows differ: {rec['groves_vs_sequential_rows_differ'].get(part)})"
                             if "groves_vs_sequential_rows_differ" in rec else ""), flush=True)
                if rec.get("error") and not rec.get("verdict"):
                    counts["REFUSED"] += 1
                    print(f"{key}: REFUSED {rec['error']}", flush=True)
    out["counts"] = counts
    _finish(args, out, witness, counts)


def _class_of(path):
    import mojolearn as ml
    arrays = ml._serialize.read_npz(path, (
        "mojolearn-randomforest-1-parallel-groves-1", "mojolearn-extratrees-1-parallel-groves-1",
        "mojolearn-randomforest-1", "mojolearn-extratrees-1"))
    return getattr(ml, ml._serialize.scalar_str(arrays, "estimator"))


def run_large(args):
    import mojolearn as ml
    with open(os.path.join(args.models_dir, "manifest.json")) as fh:
        manifest = json.load(fh)
    witness = _host_witness()
    out = dict(kind="large", vendor=ml.vendor(), numeric_mode=ml.numeric_mode(), host=witness, cells={},
               threads=os.environ.get("MOJOLEARN_CPU_THREADS", ""))
    counts = dict(IDENTICAL=0, DIVERGENT=0, REFUSED=0)
    names = [n for n in args.models.split(",") if n] if args.models else sorted(manifest["models"])
    for name in names:
        info = manifest["models"][name]
        path = os.path.join(args.models_dir, name + ".npz")
        X = np.load(os.path.join(args.models_dir, info["x"]))
        if args.rows:
            X = X[:args.rows]
        X = np.ascontiguousarray(X, dtype=np.float32)
        key = name
        t0 = time.perf_counter()
        rec = dict(rows=int(X.shape[0]), trees=info.get("trees"), depth=info.get("depth"), error=None)
        try:
            est = _class_of(path).load(path)
            if est.inference_engine != "parallel_groves":
                raise RuntimeError(f"{path} restored engine {est.inference_engine}")
            t = time.perf_counter(); gpu = _parts_of(est, X); rec["gpu_seconds"] = round(time.perf_counter() - t, 2)
            est.inference_engine = "sequential"
            seq = _parts_of(est, X)
            rec["gpu_groves"] = {k: _h(v) for k, v in gpu.items()}
            rec["sequential"] = {k: _h(v) for k, v in seq.items()}
            rec["groves_vs_sequential_rows_differ"] = {
                k: int(np.count_nonzero((gpu[k].reshape(len(gpu[k]), -1) != seq[k].reshape(len(seq[k]), -1)).any(axis=1))) for k in gpu}
            host = ml.host_model(path)
            rec["host_engine"] = host.inference_engine
            t = time.perf_counter(); hp = _parts_of(host, X); rec["host_seconds"] = round(time.perf_counter() - t, 2)
            rec["host_groves"] = {k: _h(v) for k, v in hp.items()}
            rec["verdict"] = _compare(gpu, hp)
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"
            rec["verdict"] = {"predict": f"REFUSED: {type(exc).__name__}: {exc}"}
        rec["seconds"] = round(time.perf_counter() - t0, 2)
        out["cells"][key] = rec
        for part, v in rec["verdict"].items():
            counts[v.split(":")[0]] += 1
            print(f"{key} {part}: {v} (rows {X.shape[0]}; groves vs sequential rows differ: "
                  f"{rec.get('groves_vs_sequential_rows_differ', {}).get(part)}; gpu {rec.get('gpu_seconds')}s host {rec.get('host_seconds')}s)", flush=True)
    out["counts"] = counts
    _finish(args, out, witness, counts)


def _finish(args, out, witness, counts):
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=1, default=str)
    sab = "forest-sabotage" if witness["forest_host_sabotage"] else ""
    sab += " groves-sabotage" if witness["forest_host_groves_sabotage"] else ""
    print(f"GROVES {out['kind']} host={witness['path']} sha={witness['sha256']} sabotage=[{sab.strip()}] "
          f"IDENTICAL={counts['IDENTICAL']} DIVERGENT={counts['DIVERGENT']} REFUSED={counts['REFUSED']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("lanes")
    p.add_argument("--lanes", default="", help="comma list; default every rf- and et- lane")
    p.add_argument("--fixtures", default="base,ties,odd,dupes,wide")
    p.add_argument("--json", default="")
    q = sub.add_parser("large")
    q.add_argument("--models-dir", required=True)
    q.add_argument("--models", default="", help="comma list of manifest names; default all")
    q.add_argument("--rows", type=int, default=0, help="cap the prediction rows (0: all)")
    q.add_argument("--json", default="")
    args = ap.parse_args()
    if args.cmd == "lanes":
        run_lanes(args)
    else:
        run_large(args)


if __name__ == "__main__":
    main()
