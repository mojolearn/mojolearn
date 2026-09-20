#!/usr/bin/env python3
"""Time and hash the production CPU-only IsolationForest score request."""
import argparse, hashlib, importlib.util, json, statistics, time
import numpy as np


def main():
    p = argparse.ArgumentParser(); p.add_argument("binary")
    p.add_argument("--train", type=int, default=4096); p.add_argument("--query", type=int, default=65536)
    p.add_argument("--features", type=int, default=16); p.add_argument("--trees", type=int, default=100)
    p.add_argument("--rounds", type=int, default=5); a = p.parse_args()
    spec = importlib.util.spec_from_file_location("_mojolearn_svm_host", a.binary)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    rng = np.random.default_rng(20260919)
    train = rng.standard_normal((a.train, a.features), dtype=np.float32)
    query = rng.standard_normal((a.query, a.features), dtype=np.float32)
    out = np.empty(a.query, np.float32); labels = np.empty(a.query, np.int32); info = np.empty(3, np.float64)
    # ntrain,d,nquery,trees,max_samples(auto/int/frac),max_depth,
    # max_features(int/frac),bootstrap,seed,contamination(auto/value),want(score)
    params = [a.train, a.features, a.query, a.trees, 0, 0, 0.0, -1,
              0, 0, 1.0, 0, 7, 1, 0.0, 0]
    samples, hashes = [], []
    for i in range(a.rounds + 1):
        t = time.perf_counter_ns()
        mod.iforest_run(train.ctypes.data, query.ctypes.data, out.ctypes.data,
                        labels.ctypes.data, info.ctypes.data, params)
        dt = (time.perf_counter_ns() - t) / 1e6
        if i: samples.append(dt); hashes.append(hashlib.sha256(out.tobytes()).hexdigest())
    assert len(set(hashes)) == 1
    print(json.dumps({"median_ms": statistics.median(samples), "samples_ms": samples,
                      "sha256": hashes[0], "shape": [a.train, a.query, a.features, a.trees]}, sort_keys=True))


if __name__ == "__main__": main()
