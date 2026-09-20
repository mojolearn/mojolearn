#!/usr/bin/env python3
"""Large-input host weighted-accuracy benchmark."""
import argparse
import hashlib
import importlib.util
import json
import resource
import time

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("library")
parser.add_argument("--rows", type=int, default=10_000_000)
parser.add_argument("--repeats", type=int, default=5)
args = parser.parse_args()
spec = importlib.util.spec_from_file_location("_mojolearn_metrics_host", args.library)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
rng = np.random.default_rng(20260920)
y = rng.integers(0, 11, size=args.rows, dtype=np.int32)
p = ((y + rng.integers(0, 4, size=args.rows, dtype=np.int32)) % 11).astype(np.int32)
w = rng.random(args.rows, dtype=np.float32)
h = hashlib.sha256()
for array in (y, p, w):
    h.update(memoryview(array).cast("B"))
times, values = [], []
for _ in range(args.repeats):
    start = time.perf_counter()
    values.append(float(module.accuracy_score_weighted(
        y.ctypes.data, p.ctypes.data, w.ctypes.data, [args.rows])))
    times.append(time.perf_counter() - start)
errors = []
for count, weights in ((0, w), (args.rows, np.zeros_like(w))):
    try:
        module.accuracy_score_weighted(y.ctypes.data, p.ctypes.data,
                                       weights.ctypes.data, [count])
    except Exception as exc:
        errors.append(str(exc))
print(json.dumps({"rows": args.rows, "seed": 20260920,
                  "input_sha256": h.hexdigest(), "seconds": times,
                  "values": values, "errors": errors,
                  "maxrss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}))
