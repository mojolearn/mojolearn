#!/usr/bin/env python3
"""Alternating exact StandardScaler.fit_transform timing on one full block.

Each arm runs from an isolated Python tree.  The candidate tree must expose
the experimental fused native ABI; the baseline tree must not.  One warmup is
discarded, then five alternating public fit_transform calls are timed.  Every
fitted statistic and every transformed output byte must agree across arms and
repeats before a speed result can pass.
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


def digest(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(value.dtype.str.encode())
    h.update(str(value.shape).encode())
    h.update(value.data)
    return h.hexdigest()


def worker(args):
    protocol = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    os.environ["MOJOLEARN_NUMERIC_MODE"] = "identical"
    import mojolearn as ml

    with np.load(args.block) as loaded:
        X = np.ascontiguousarray(loaded["X"], dtype=np.float32)
    probe = ml.StandardScaler()
    binding = probe._binding("identical")
    fused = hasattr(binding, "standard_fit_transform")
    if fused != (args.arm == "fused"):
        raise RuntimeError("native fused ABI reach mismatch: %s" % fused)

    def emit(record):
        protocol.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")

    emit({"event": "ready", "arm": args.arm, "fused_native": fused,
          "shape": list(X.shape), "dtype": X.dtype.str})
    for line in sys.stdin:
        command, _, number = line.strip().partition(" ")
        if command == "quit":
            emit({"event": "bye"})
            return 0
        if command != "run":
            raise RuntimeError("unknown command %r" % command)
        seq = int(number)
        model = ml.StandardScaler(with_mean=True, with_std=True,
                                  numeric_mode="identical")
        start = time.perf_counter()
        output = np.asarray(model.fit_transform(X))
        elapsed = 1000.0 * (time.perf_counter() - start)
        stats = np.stack((np.asarray(model.mean_), np.asarray(model.var_),
                          np.asarray(model.scale_)))
        quality = {
            "finite": bool(np.isfinite(output).all() and np.isfinite(stats).all()),
            "variance_nonnegative": bool(np.all(stats[1] >= 0)),
            "scale_positive": bool(np.all(stats[2] > 0)),
            "samples": int(model.n_samples_seen_),
            "features": int(model.n_features_in_),
        }
        emit({"event": "run", "seq": seq, "ms": elapsed,
              "stats_sha256": digest(stats), "output_sha256": digest(output),
              "quality": quality})


class Child:
    def __init__(self, name, python, tree, block, log):
        python = os.path.abspath(python)
        tree = os.path.abspath(tree)
        env = dict(os.environ)
        env["PYTHONPATH"] = os.path.join(tree, "python")
        env["MOJOLEARN_NUMERIC_MODE"] = "identical"
        self.name = name
        self.log = open(log, "w")
        self.proc = subprocess.Popen(
            [python, os.path.abspath(__file__), "worker", "--arm", name,
             "--block", block], cwd=tree, env=env, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self.log, start_new_session=True)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    w = sub.add_parser("worker")
    w.add_argument("--arm", choices=("off", "fused"), required=True)
    w.add_argument("--block", required=True)
    r = sub.add_parser("race")
    r.add_argument("--block", required=True)
    r.add_argument("--arm", action="append", nargs=3,
                   metavar=("NAME", "PYTHON", "TREE"), required=True)
    r.add_argument("--rounds", type=int, default=5)
    r.add_argument("--timeout", type=float, default=1800)
    r.add_argument("--spread-gate", type=float, default=1.10)
    r.add_argument("--min-speedup", type=float, default=1.02)
    r.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "worker":
        return worker(args)
    arms = {name: (python, tree) for name, python, tree in args.arm}
    if set(arms) != {"off", "fused"} or len(args.arm) != 2 or args.rounds < 1:
        parser.error("race requires exactly off and fused arms and positive rounds")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    children = {}
    records = {name: [] for name in arms}
    ready = {}
    try:
        for name, (python, tree) in arms.items():
            children[name] = Child(name, python, tree, args.block,
                                   args.output + "." + name + ".log")
            ready[name] = children[name].read(args.timeout)
        # Warm each independently. Alternate first arm every measured round.
        for name in ("off", "fused"):
            children[name].send("run -1")
            children[name].read(args.timeout)
        for seq in range(args.rounds):
            order = ("off", "fused") if seq % 2 == 0 else ("fused", "off")
            for name in order:
                children[name].send("run %d" % seq)
                record = children[name].read(args.timeout)
                if record.get("event") != "run" or record.get("seq") != seq:
                    raise RuntimeError("unexpected record %r" % record)
                records[name].append(record)
    finally:
        for child in children.values():
            child.close()

    all_records = records["off"] + records["fused"]
    for field in ("stats_sha256", "output_sha256", "quality"):
        values = {json.dumps(row[field], sort_keys=True) for row in all_records}
        if len(values) != 1:
            raise RuntimeError("bitwise/quality mismatch in %s: %s" % (field, values))
    quality = all_records[0]["quality"]
    if not all((quality["finite"], quality["variance_nonnegative"],
                quality["scale_positive"])):
        raise RuntimeError("invalid scaler quality: %r" % quality)
    samples = {name: [row["ms"] for row in rows]
               for name, rows in records.items()}
    medians = {name: statistics.median(values)
               for name, values in samples.items()}
    spreads = {name: max(values) / min(values)
               for name, values in samples.items()}
    speedup = medians["off"] / medians["fused"]
    eligible = (speedup >= args.min_speedup and
                all(value <= args.spread_gate for value in spreads.values()))
    result = {"block": os.path.abspath(args.block), "ready": ready,
              "arms": {name: {"samples_ms": samples[name],
                               "median_ms": medians[name],
                               "spread": spreads[name]}
                       for name in arms},
              "speedup": speedup, "quality": quality,
              "stats_sha256": all_records[0]["stats_sha256"],
              "output_sha256": all_records[0]["output_sha256"],
              "promotion_eligible": eligible,
              "verdict": ("PROMOTION_ELIGIBLE_BITWISE_AND_QUALITY_EQUAL"
                          if eligible else "REJECT_TIMING_GATE")}
    with open(args.output, "x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print("STANDARD-FIT-TRANSFORM", result["verdict"],
          "speedup=%.6fx" % speedup, "spread=%r" % spreads)
    return 0 if eligible else 1


if __name__ == "__main__":
    raise SystemExit(main())
