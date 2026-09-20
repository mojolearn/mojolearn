#!/usr/bin/env python3
"""Time and hash one production host GaussianMixture scoring call."""
import argparse
import hashlib
import importlib.util
import json
import statistics
import time

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("binary")
    p.add_argument("api", choices=("predict", "score_samples", "predict_proba"))
    p.add_argument("--rows", type=int, default=32768)
    p.add_argument("--features", type=int, default=24)
    p.add_argument("--components", type=int, default=8)
    p.add_argument("--rounds", type=int, default=7)
    a = p.parse_args()
    spec = importlib.util.spec_from_file_location("_mojolearn_mixture_host", a.binary)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rng = np.random.default_rng(20260919)
    n, d, k = a.rows, a.features, a.components
    x = rng.standard_normal((n, d), dtype=np.float32)
    means = rng.standard_normal((k, d), dtype=np.float32)
    precision = np.zeros((k, d, d), dtype=np.float32)
    for c in range(k):
        precision[c] = np.triu(rng.standard_normal((d, d), dtype=np.float32) * np.float32(.03))
        precision[c].flat[::d + 1] += np.float32(1)
    weights = np.full(k, np.float32(1 / k), dtype=np.float32)
    log_det = np.log(np.diagonal(precision, axis1=1, axis2=2)).sum(1).astype(np.float32)
    covariance = np.empty_like(precision)
    dtype, size = (np.int32, n) if a.api == "predict" else (np.float32, n * k if a.api == "predict_proba" else n)
    out = np.empty(size, dtype=dtype)
    addrs = [z.ctypes.data for z in (weights, means, covariance, precision, log_det, x, out)]
    params = [k, d, 0, 1, 0.0, n]
    call = getattr(mod, "gmm_" + a.api)
    samples, hashes = [], []
    for i in range(a.rounds + 2):
        t = time.perf_counter_ns(); call(addrs, params); dt = (time.perf_counter_ns() - t) / 1e6
        if i >= 2:
            samples.append(dt); hashes.append(hashlib.sha256(out.tobytes()).hexdigest())
    assert len(set(hashes)) == 1
    print(json.dumps({"api": a.api, "median_ms": statistics.median(samples),
                      "samples_ms": samples, "sha256": hashes[0], "shape": [n, d, k]}, sort_keys=True))


if __name__ == "__main__":
    main()
