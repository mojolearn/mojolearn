#!/usr/bin/env python3
"""Interleaved AMD KMeans fit/transform A/B on a prepared CTD dataset.

Each arm is an isolated Python tree containing its own compiled binding.  The
public ``fit`` and ``transform`` calls are timed; hashing and quality checks
are outside those clocks.  The conductor refuses any byte difference between
the arms or between repeated calls.
"""
import argparse
import hashlib
import json
import os
import select
import signal
import statistics
import subprocess
import sys
import time

import numpy as np


def digest_array(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(value.dtype.str.encode())
    h.update(str(value.shape).encode())
    h.update(value.data)
    return h.hexdigest()


def fit_snapshot(est):
    arrays = {
        "centers": np.asarray(est.cluster_centers_),
        "labels": np.asarray(est.labels_),
        "n_iter": np.asarray([est.n_iter_], dtype=np.int64),
    }
    for name in ("inertia_", "sum_scale_", "weight_scale_"):
        if hasattr(est, name):
            arrays[name[:-1]] = np.asarray([getattr(est, name)])
    quality = {
        "rows": int(arrays["labels"].size),
        "clusters": int(arrays["centers"].shape[0]),
        "finite_centers": bool(np.isfinite(arrays["centers"]).all()),
        "labels_in_range": bool(
            np.all((arrays["labels"] >= 0) &
                   (arrays["labels"] < arrays["centers"].shape[0]))),
        "n_iter": int(est.n_iter_),
    }
    if hasattr(est, "inertia_"):
        quality["inertia"] = float(est.inertia_)
    return {
        "parts": {name: digest_array(value) for name, value in arrays.items()},
        "quality": quality,
    }


def worker(args):
    protocol = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    def emit(record):
        protocol.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")

    os.environ["MOJOLEARN_NUMERIC_MODE"] = "identical"
    import mojolearn as ml

    with np.load(args.block) as loaded:
        X = np.ascontiguousarray(loaded["X"])
        Xq = np.ascontiguousarray(loaded["Xq"][:args.transform_rows])
        init = np.ascontiguousarray(loaded["init"])
    model = None

    def new_model():
        return ml.KMeans(n_clusters=64, init="array", init_centroids=init,
                         n_init=1, max_iter=20, tol=1e-7, random_state=0,
                         metric="euclidean")

    probe = new_model()
    vendor = probe.vendor_used()
    mode = probe.numeric_mode_used()
    if vendor != "hip" or mode != "identical":
        raise RuntimeError("expected hip/identical, got %s/%s" % (vendor, mode))
    emit({"event": "ready", "vendor": vendor, "numeric_mode": mode,
          "fit_shape": list(X.shape), "transform_shape": [len(Xq), 64]})
    for line in sys.stdin:
        command, _, number = line.partition(" ")
        seq = int(number or 0)
        if command == "fit":
            start = time.perf_counter()
            model = new_model().fit(X)
            elapsed = 1000.0 * (time.perf_counter() - start)
            snap = fit_snapshot(model)
            if not snap["quality"]["finite_centers"] or not snap["quality"]["labels_in_range"]:
                raise RuntimeError("invalid fitted model quality: %r" % snap["quality"])
            emit({"event": "fit", "seq": seq, "ms": elapsed, **snap})
        elif command == "transform":
            if model is None:
                raise RuntimeError("transform requested before fit")
            start = time.perf_counter()
            distances = np.asarray(model.transform(Xq))
            elapsed = 1000.0 * (time.perf_counter() - start)
            prediction = np.asarray(model.predict(Xq), dtype=np.int64)
            argmin = np.argmin(distances, axis=1).astype(np.int64, copy=False)
            selected = distances[np.arange(len(prediction)), prediction]
            minima = distances.min(axis=1)
            emit({"event": "transform", "seq": seq, "ms": elapsed,
                  "digest": digest_array(distances),
                  "predict_digest": digest_array(prediction),
                  "quality": {"finite": bool(np.isfinite(distances).all()),
                              "argmin_agreement_fraction": float(np.mean(prediction == argmin)),
                              "selected_equals_min": bool(np.array_equal(selected, minima))}})
        elif command == "quit":
            emit({"event": "bye"})
            return 0
        else:
            raise RuntimeError("unknown command %r" % command)
    return 0


class Child:
    def __init__(self, name, python, tree, args, log):
        env = dict(os.environ)
        env["PYTHONPATH"] = os.path.join(tree, "python")
        env["MOJOLEARN_NUMERIC_MODE"] = "identical"
        cmd = [python, os.path.abspath(__file__), "worker", "--block", args.block,
               "--transform-rows", str(args.transform_rows)]
        self.name = name
        self.log = open(log, "w")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self.log, env=env, cwd=tree,
                                     start_new_session=True)
        self.buffer = b""

    def send(self, command):
        self.proc.stdin.write((command + "\n").encode())
        self.proc.stdin.flush()

    def read(self, timeout):
        deadline = time.monotonic() + timeout
        fd = self.proc.stdout.fileno()
        while b"\n" not in self.buffer:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError("%s worker timed out" % self.name)
            ready, _, _ = select.select([fd], [], [], min(left, 5.0))
            if ready:
                chunk = os.read(fd, 65536)
                if not chunk:
                    raise RuntimeError("%s worker exited rc=%s" %
                                       (self.name, self.proc.poll()))
                self.buffer += chunk
        line, self.buffer = self.buffer.split(b"\n", 1)
        return json.loads(line)

    def close(self):
        try:
            self.send("quit")
            self.read(60)
        except Exception:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except OSError:
                pass
        self.proc.wait(timeout=60)
        self.log.close()


def same(records, field, label):
    values = [json.dumps(record[field], sort_keys=True) for record in records]
    if len(set(values)) != 1:
        raise RuntimeError("%s differs: %s" % (label, values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    w = sub.add_parser("worker")
    w.add_argument("--block", required=True)
    w.add_argument("--transform-rows", type=int, required=True)
    r = sub.add_parser("race")
    r.add_argument("--block", required=True)
    r.add_argument("--arm", action="append", nargs=3, metavar=("NAME", "PYTHON", "TREE"),
                   required=True)
    r.add_argument("--rounds", type=int, default=5)
    r.add_argument("--transform-rounds", type=int, default=7)
    r.add_argument("--transform-rows", type=int, default=100000)
    r.add_argument("--timeout", type=float, default=900)
    r.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "worker":
        return worker(args)
    if len(args.arm) < 2 or args.rounds < 1 or args.transform_rounds < 1:
        parser.error("race requires at least two arms and positive round counts")
    if args.transform_rows < 1:
        parser.error("--transform-rows must be positive")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    children = {}
    result = {"block": os.path.abspath(args.block), "arms": {},
              "rounds": args.rounds, "transform_rounds": args.transform_rounds,
              "transform_rows_requested": args.transform_rows}
    try:
        for name, python, tree in args.arm:
            if name in children:
                parser.error("duplicate arm %s" % name)
            child = Child(name, python, tree, args, args.output + "." + name + ".log")
            children[name] = child
            result["arms"][name] = {"ready": child.read(args.timeout),
                                     "fit": [], "transform": []}
        names = list(children)
        # Round zero warms both arms through the exact public path.
        for phase, count in (("fit", args.rounds + 1),
                             ("transform", args.transform_rounds + 1)):
            for seq in range(count):
                round_records = []
                order = names[seq % len(names):] + names[:seq % len(names)]
                for name in order:
                    child = children[name]
                    child.send("%s %d" % (phase, seq))
                    record = child.read(args.timeout)
                    if record.get("event") != phase:
                        raise RuntimeError("unexpected worker record %r" % record)
                    if seq:
                        result["arms"][name][phase].append(record)
                    round_records.append(record)
                if phase == "fit":
                    same(round_records, "parts", "fit bytes in round %d" % seq)
                    same(round_records, "quality", "fit quality in round %d" % seq)
                else:
                    same(round_records, "digest", "transform bytes in round %d" % seq)
                    same(round_records, "predict_digest", "predict bytes in round %d" % seq)
                    same(round_records, "quality", "transform quality in round %d" % seq)
                    quality = round_records[0]["quality"]
                    if not quality["finite"] or not quality["selected_equals_min"]:
                        raise RuntimeError("transform quality failed: %r" % quality)
        # Repeated calls in each arm must also be byte stable.
        for name, arm in result["arms"].items():
            same(arm["fit"], "parts", name + " repeated fit")
            same(arm["transform"], "digest", name + " repeated transform")
            for phase in ("fit", "transform"):
                samples = [row["ms"] for row in arm[phase]]
                arm[phase + "_median_ms"] = statistics.median(samples)
        baseline, candidate = names[:2]
        for phase in ("fit", "transform"):
            b = result["arms"][baseline][phase + "_median_ms"]
            c = result["arms"][candidate][phase + "_median_ms"]
            result[phase + "_speedup"] = b / c
        result["verdict"] = "BITWISE_IDENTICAL_AND_QUALITY_EQUAL"
        with open(args.output, "x") as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        print("KMEANS-AMD", result["verdict"],
              "fit_speedup=%.3f" % result["fit_speedup"],
              "transform_speedup=%.3f" % result["transform_speedup"])
        return 0
    finally:
        for child in children.values():
            child.close()


if __name__ == "__main__":
    sys.exit(main())
