#!/usr/bin/env python3
"""Reproducible >=1M-row benchmark for host split-K Gram consumers."""
import argparse
import hashlib
import importlib.util
import json
import platform
import resource
import time

import numpy as np


def load(path):
    spec = importlib.util.spec_from_file_location("_mojolearn_estimators_host", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(*arrays):
    h = hashlib.sha256()
    for array in arrays:
        h.update(np.asarray(array).tobytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("library")
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--cols", type=int, default=16)
    parser.add_argument("--components", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    module = load(args.library)
    rng = np.random.default_rng(20260920)
    x = rng.standard_normal((args.rows, args.cols), dtype=np.float32)
    y = rng.standard_normal(args.rows, dtype=np.float32)

    def pca():
        comp = np.empty((args.components, args.cols), np.float32)
        mean = np.empty(args.cols, np.float32)
        ev = np.empty(args.components, np.float32)
        ratio = np.empty(args.components, np.float32)
        sv = np.empty(args.components, np.float32)
        noise = module.pca_fit(x.ctypes.data, comp.ctypes.data, mean.ctypes.data,
                               ev.ctypes.data, ratio.ctypes.data, sv.ctypes.data,
                               [args.rows, args.cols, args.components])
        return digest(comp, mean, ev, ratio, sv, np.float64(noise))

    def tsvd():
        comp = np.empty((args.components, args.cols), np.float32)
        sv = np.empty(args.components, np.float32)
        module.tsvd_fit(x.ctypes.data, comp.ctypes.data, sv.ctypes.data,
                        [args.rows, args.cols, args.components])
        return digest(comp, sv)

    def ols():
        coef = np.empty(args.cols, np.float32)
        module.ols_fit(x.ctypes.data, y.ctypes.data, coef.ctypes.data,
                       [args.rows, args.cols])
        return digest(coef)

    results = {}
    for name, fn in (("pca", pca), ("tsvd", tsvd), ("ols", ols)):
        samples, hashes = [], []
        for _ in range(args.repeats):
            start = time.perf_counter()
            hashes.append(fn())
            samples.append(time.perf_counter() - start)
        results[name] = {"seconds": samples, "hashes": hashes}
    results.update(rows=args.rows, cols=args.cols, seed=20260920,
                   input_sha256=digest(x, y), machine=platform.machine(),
                   maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    print(json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()
