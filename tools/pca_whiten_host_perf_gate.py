#!/usr/bin/env python3
"""Check CPU PCA whitening identity and speed against a baseline binary."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
import subprocess
import sys
import time

SHAPES = ((100_000, 8, 4), (20_000, 32, 16))


def _worker(path: str, repeats: int) -> list[dict]:
    import numpy as np

    spec = importlib.util.spec_from_file_location("_mojolearn_estimators_host", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = []
    for nr, nf, nc in SHAPES:
        rng = np.random.default_rng(23)
        x = rng.normal(size=(nr, nf)).astype(np.float32)
        mean = rng.normal(size=nf).astype(np.float32)
        components = rng.normal(size=(nc, nf)).astype(np.float32)
        singular = (rng.random(nc) + 0.1).astype(np.float32)
        out = np.empty((nr, nc), np.float32)
        forward_args = (x.ctypes.data, mean.ctypes.data, components.ctypes.data,
                        singular.ctypes.data, out.ctypes.data, [nr, nf, nc, 4096])
        module.pca_whiten_transform(*forward_args)
        forward = []
        for _ in range(repeats):
            start = time.perf_counter()
            module.pca_whiten_transform(*forward_args)
            forward.append(time.perf_counter() - start)
        scores = out.copy()
        restored = np.empty((nr, nf), np.float32)
        inverse_args = (scores.ctypes.data, components.ctypes.data,
                        singular.ctypes.data, mean.ctypes.data,
                        restored.ctypes.data, [nr, nf, nc, 4096])
        module.pca_whiten_inverse_transform(*inverse_args)
        inverse = []
        for _ in range(repeats):
            start = time.perf_counter()
            module.pca_whiten_inverse_transform(*inverse_args)
            inverse.append(time.perf_counter() - start)
        rows.append({
            "shape": [nr, nf, nc], "forward_s": forward, "inverse_s": inverse,
            "forward_sha256": hashlib.sha256(out.tobytes()).hexdigest(),
            "inverse_sha256": hashlib.sha256(restored.tobytes()).hexdigest(),
        })
    return rows


def _run(path: str, repeats: int) -> list[dict]:
    code = ("import json,sys; from tools.pca_whiten_host_perf_gate import _worker; "
            "print(json.dumps(_worker(sys.argv[1], int(sys.argv[2]))))")
    proc = subprocess.run([sys.executable, "-c", code, path, str(repeats)],
                          check=True, capture_output=True, text=True)
    return json.loads(proc.stdout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--min-speedup", type=float, default=1.0)
    args = parser.parse_args()
    if args.repeats < 1 or args.min_speedup <= 0:
        parser.error("--repeats and --min-speedup must be positive")
    baseline, candidate = _run(args.baseline, args.repeats), _run(args.candidate, args.repeats)
    report = []
    for before, after in zip(baseline, candidate, strict=True):
        if before["shape"] != after["shape"]:
            raise AssertionError("shape mismatch")
        for direction in ("forward", "inverse"):
            hash_key = f"{direction}_sha256"
            if before[hash_key] != after[hash_key]:
                raise AssertionError(f"{direction} bits moved at {before['shape']}")
            base_s = statistics.median(before[f"{direction}_s"])
            new_s = statistics.median(after[f"{direction}_s"])
            speedup = base_s / new_s
            if speedup < args.min_speedup:
                raise AssertionError(
                    f"{direction} speedup {speedup:.3f}x at {before['shape']} "
                    f"is below {args.min_speedup:.3f}x")
            report.append({"shape": before["shape"], "direction": direction,
                           "baseline_s": base_s, "candidate_s": new_s,
                           "speedup": speedup, "sha256": before[hash_key]})
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
