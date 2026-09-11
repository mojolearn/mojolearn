#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""UMAP on the two datasets: OUR IDENTICAL fit_transform beside cuML UMAP,
interleaved round by round, trustworthiness beside time (lane/knn-speed,
2026-09-11; ENGINEERING_RULES.md section 9).

    python3 tools/umap_two_datasets.py race --dataset taxi --rows 100000 \\
        --data /root/ctd-data --out DIR [--rounds 3] [--arms ours,cuml-gpu]

THE DATA is the kNN block `tools/classical_two_datasets.py prep --lanes knn`
writes (`<data>/knn-<dataset>.npz`, key `index`: NYC taxi's 11 numeric
columns or Istella-S's 220 features, rows [0, 400,000) of the trees
harness's cache, Istella's float32-max sentinel already replaced by 0.0).
UMAP fits the leading `--rows` rows, raw columns, no scaling.

THE ARMS
  ours      mojolearn.UMAP(n_neighbors=15, n_components=2, n_epochs=200,
            random_state=0) under MOJOLEARN_NUMERIC_MODE=identical, host
            float32 in, host embedding out, the public call's upload inside
            the clock.
  cuml-gpu  cuml.manifold.UMAP(n_neighbors=15, n_components=2, n_epochs=200,
            init='spectral', output_type='cupy'), random_state unset (their
            FAST arm: a seed makes cuML's optimizer take its reproducible,
            slower path), input uploaded as cupy before the clock, the clock
            ends at cupy.cuda.runtime.deviceSynchronize().

INTERLEAVING: one persistent worker process per arm; round 0 is a warm-up and
excluded, rounds 1..--rounds are timed, the arm order rotates every round.

QUALITY, computed here from each arm's last embedding, float64 NumPy, never
a library scorer: trustworthiness and 10-neighbor retention on a stride
sample of --quality-rows rows (exact ranks inside the sample; the same
sample for every arm). A sampled trustworthiness is the score of the
sample's own neighborhoods and is labeled so.

OUTPUT: <out>/umap-<dataset>.json (every round, embedding sha256 per round,
quality, ratio ours median / cuML median), worker logs beside it.
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

PREFIX = "UTD "
NEIGHBORS = 15
EPOCHS = 200


def load_rows(data, dataset, rows):
    path = os.path.join(data, "knn-%s.npz" % dataset)
    with np.load(path) as z:
        x = np.ascontiguousarray(z["index"][:rows], dtype=np.float32)
    if x.shape[0] != rows:
        raise SystemExit("%s holds %d rows, asked for %d" % (path, x.shape[0], rows))
    return x, path


def say(msg):
    sys.__stdout__.write(PREFIX + msg + "\n")
    sys.__stdout__.flush()


def worker(args):
    x, path = load_rows(args.data, args.dataset, args.rows)
    info = {"arm": args.arm, "rows": int(x.shape[0]), "features": int(x.shape[1]),
            "block": path, "sha256_x": hashlib.sha256(x.tobytes()).hexdigest()}
    if args.arm == "ours":
        os.environ.setdefault("MOJOLEARN_NUMERIC_MODE", "identical")
        import mojolearn
        info["library"] = "mojolearn %s" % getattr(mojolearn, "__version__", "?")

        def fit():
            model = mojolearn.UMAP(n_neighbors=NEIGHBORS, n_components=2,
                                   n_epochs=EPOCHS, random_state=0)
            emb = np.array(model.fit_transform(x), dtype=np.float32)
            try:
                info["numeric_mode_used"] = model.numeric_mode_used()
            except Exception as exc:  # noqa: BLE001
                info["numeric_mode_used"] = "unavailable (%r)" % (exc,)
            return emb

        def sync():
            pass
    elif args.arm == "cuml-gpu":
        import cupy as cp
        import cuml
        from cuml.manifold import UMAP
        cp.cuda.runtime.deviceSynchronize()
        gx = cp.asarray(x)
        cp.cuda.runtime.deviceSynchronize()
        info["library"] = "cuml %s cupy %s" % (cuml.__version__, cp.__version__)
        holder = {}

        def fit():
            model = UMAP(n_neighbors=NEIGHBORS, n_components=2, n_epochs=EPOCHS,
                         init="spectral", output_type="cupy")
            holder["e"] = model.fit_transform(gx)
            return None

        def sync():
            cp.cuda.runtime.deviceSynchronize()
    else:
        raise SystemExit("unknown arm %r" % (args.arm,))
    say("READY " + json.dumps(info))
    for line in sys.stdin:
        cmd = line.split()
        if not cmd or cmd[0] == "quit":
            break
        r = int(cmd[1])
        t0 = time.perf_counter()
        emb = fit()
        sync()
        ms = (time.perf_counter() - t0) * 1000.0
        if emb is None:
            emb = np.asarray(holder["e"].get(), dtype=np.float32)
        if emb.shape != (x.shape[0], 2) or not np.isfinite(emb).all():
            say("FAIL %d invalid embedding %r" % (r, emb.shape))
            continue
        np.save(os.path.join(args.work, "%s-%s.npy" % (args.arm, args.dataset)), emb)
        say("DONE %d %.6f %s %s" % (r, ms, hashlib.sha256(emb.tobytes()).hexdigest(),
                                   json.dumps({"numeric_mode_used": info.get("numeric_mode_used")})))


def trust(x, emb, k=10):
    """Trustworthiness and k-neighbor retention, exact ranks, stable ties
    (`tools/nvidia_public_compare.py::neighborhood_quality`'s formula)."""
    n = len(x)

    def order(v):
        v = np.asarray(v, dtype=np.float64)
        sq = (v * v).sum(axis=1)
        d = sq[:, None] + sq[None, :] - 2.0 * (v @ v.T)
        np.fill_diagonal(d, np.inf)
        return np.argsort(d, axis=1, kind="stable")

    original, reduced = order(x), order(emb)
    ranks = np.empty((n, n), dtype=np.int32)
    ranks[np.arange(n)[:, None], original] = np.arange(1, n + 1)
    nr = ranks[np.arange(n)[:, None], reduced[:, :k]]
    penalty = np.maximum(nr - k, 0).sum(dtype=np.int64)
    return {"trustworthiness": 1.0 - 2.0 * float(penalty) / (n * k * (2 * n - 3 * k - 1)),
            "neighbor_retention": float((nr <= k).mean()), "k": k, "sample_rows": n}


def read_proto(proc, deadline):
    while True:
        if time.time() > deadline:
            raise TimeoutError("worker deadline")
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("worker exited (rc %r)" % (proc.poll(),))
        if line.startswith(PREFIX):
            return line[len(PREFIX):].rstrip("\n")


def race(args):
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.work, exist_ok=True)
    arms = [a for a in args.arms.split(",") if a]
    procs, ready = {}, {}
    for arm in arms:
        log = open(os.path.join(args.out, "umap-%s-%s.log" % (args.dataset, arm)), "w")
        cmd = [sys.executable, os.path.abspath(__file__), "worker", "--arm", arm,
               "--dataset", args.dataset, "--rows", str(args.rows), "--data", args.data,
               "--work", args.work]
        procs[arm] = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=log, text=True, bufsize=1)
        msg = read_proto(procs[arm], time.time() + args.round_seconds)
        if not msg.startswith("READY "):
            raise RuntimeError("%s: %s" % (arm, msg))
        ready[arm] = json.loads(msg[6:])
        print("UTD-READY", arm, json.dumps(ready[arm]), flush=True)
    rounds = {a: [] for a in arms}
    for r in range(args.rounds + 1):
        order = arms[r % len(arms):] + arms[:r % len(arms)]
        for arm in order:
            procs[arm].stdin.write("round %d\n" % r)
            procs[arm].stdin.flush()
            msg = read_proto(procs[arm], time.time() + args.round_seconds)
            parts = msg.split(" ", 4)
            if parts[0] != "DONE":
                raise RuntimeError("%s round %d: %s" % (arm, r, msg))
            rec = {"round": r, "ms": float(parts[2]), "sha256": parts[3], "warmup": r == 0}
            rounds[arm].append(rec)
            print("UTD-ROUND dataset=%s arm=%s round=%d ms=%.1f sha=%s" % (
                args.dataset, arm, r, rec["ms"], rec["sha256"][:16]), flush=True)
            time.sleep(args.pause)
    for arm in arms:
        procs[arm].stdin.write("quit\n")
        procs[arm].stdin.flush()
        procs[arm].wait(timeout=120)
    x, path = load_rows(args.data, args.dataset, args.rows)
    stride = max(1, args.rows // args.quality_rows)
    idx = np.arange(0, args.rows, stride)[:args.quality_rows]
    result = {"dataset": args.dataset, "rows": args.rows, "block": path,
              "neighbors": NEIGHBORS, "epochs": EPOCHS, "quality_sample": "stride %d, %d rows" % (stride, len(idx)),
              "ready": ready, "arms": {}}
    for arm in arms:
        timed = [t["ms"] for t in rounds[arm] if not t["warmup"]]
        emb = np.load(os.path.join(args.work, "%s-%s.npy" % (arm, args.dataset)))
        q = trust(x[idx], emb[idx], 10)
        shas = sorted({t["sha256"] for t in rounds[arm]})
        result["arms"][arm] = {"rounds": rounds[arm], "median_ms": statistics.median(timed),
                               "min_ms": min(timed), "max_ms": max(timed),
                               "digest_stable": len(shas) == 1, "quality": q}
        print("UTD-RESULT dataset=%s rows=%d arm=%s median_ms=%.1f range=%.1f..%.1f digest_stable=%s trust=%.6f retention=%.6f" % (
            args.dataset, args.rows, arm, result["arms"][arm]["median_ms"], min(timed), max(timed),
            len(shas) == 1, q["trustworthiness"], q["neighbor_retention"]), flush=True)
    if "ours" in result["arms"] and "cuml-gpu" in result["arms"]:
        result["ours_over_cuml"] = result["arms"]["ours"]["median_ms"] / result["arms"]["cuml-gpu"]["median_ms"]
        print("UTD-RATIO dataset=%s ours_over_cuml=%.3f" % (args.dataset, result["ours_over_cuml"]), flush=True)
    try:
        result["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                                       capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        result["gpu"] = "unavailable (%r)" % (exc,)
    with open(os.path.join(args.out, "umap-%s.json" % args.dataset), "w") as fh:
        json.dump(result, fh, indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("race", "worker"):
        p = sub.add_parser(name)
        p.add_argument("--dataset", required=True, choices=("taxi", "istella"))
        p.add_argument("--rows", type=int, default=100000)
        p.add_argument("--data", default="/root/ctd-data")
        p.add_argument("--work", default="/root/umap-work")
        if name == "race":
            p.add_argument("--out", required=True)
            p.add_argument("--arms", default="ours,cuml-gpu")
            p.add_argument("--rounds", type=int, default=3)
            p.add_argument("--quality-rows", type=int, default=4000)
            p.add_argument("--round-seconds", type=float, default=900.0)
            p.add_argument("--pause", type=float, default=0.5)
        else:
            p.add_argument("--arm", required=True)
    args = ap.parse_args()
    if args.cmd == "worker":
        worker(args)
    else:
        race(args)


if __name__ == "__main__":
    main()
