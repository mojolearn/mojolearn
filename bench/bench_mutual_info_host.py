#!/usr/bin/env python3
"""Large-label host mutual-information benchmark."""
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
parser.add_argument("--repeats", type=int, default=7)
args = parser.parse_args()
spec = importlib.util.spec_from_file_location("_mojolearn_metrics_host", args.library)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
rng = np.random.default_rng(20260920)
y = rng.integers(0, 32, size=args.rows, dtype=np.int32)
p = ((y.astype(np.int64) * 7 + rng.integers(0, 9, size=args.rows)) % 32).astype(np.int32)
input_hash = hashlib.sha256()
input_hash.update(memoryview(y).cast("B"))
input_hash.update(memoryview(p).cast("B"))
input_sha256 = input_hash.hexdigest()
times, values = [], []
for _ in range(args.repeats):
    start = time.perf_counter()
    values.append(float(module.mutual_info_score(y.ctypes.data, p.ctypes.data,
                                                 [args.rows, 0, 31])))
    times.append(time.perf_counter() - start)
errors = []
for params in ([0, 0, 31], [args.rows, 4, 3]):
    try:
        module.mutual_info_score(y.ctypes.data, p.ctypes.data, params)
    except Exception as exc:
        errors.append(str(exc))
print(json.dumps({"rows": args.rows, "seed": 20260920,
                  "input_sha256": input_sha256, "seconds": times,
                  "values": values, "errors": errors,
                  "maxrss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}))
