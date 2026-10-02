#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""GEMM host-surface transport probe (lane/gap-neural-models, 2026-10-02).

Times `mojolearn.linalg.matmul` / `matmul_bf16` at the board shape under each
transport arm of gemm/host_transport.mojo (env read per call), the way the
board clocks it (host arrays in, result on the host). Prints per arm the
median wall ms, the output sha (every arm must print the same one: the
transport moves no bit). GPU path only; run on the boxes, never the laptop.

    python3 tools/probe_gemm_host.py [--n 4096] [--reps 5] [--lanes gemm,gemm-bf16]
"""
import argparse
import hashlib
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "python"))

ARMS = {
    "old":       {"MOJOLEARN_GEMM_POOL": "0", "MOJOLEARN_GEMM_STAGE_DOWN": "0", "MOJOLEARN_GEMM_STAGE_UP": "0"},
    "pool":      {"MOJOLEARN_GEMM_POOL": "1", "MOJOLEARN_GEMM_STAGE_DOWN": "0", "MOJOLEARN_GEMM_STAGE_UP": "0"},
    "pool+down": {"MOJOLEARN_GEMM_POOL": "1", "MOJOLEARN_GEMM_STAGE_DOWN": "1", "MOJOLEARN_GEMM_STAGE_UP": "0"},
    "pool+both": {"MOJOLEARN_GEMM_POOL": "1", "MOJOLEARN_GEMM_STAGE_DOWN": "1", "MOJOLEARN_GEMM_STAGE_UP": "1"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--lanes", default="gemm,gemm-bf16")
    args = ap.parse_args()
    import numpy as np
    import mojolearn.linalg as linalg
    rng = np.random.default_rng(7)
    n = args.n
    a = rng.standard_normal((n, n), dtype=np.float32)
    b = rng.standard_normal((n, n), dtype=np.float32)
    for lane in args.lanes.split(","):
        if lane == "gemm":
            fn, x, y = linalg.matmul, a, b
        else:
            fn, x, y = linalg.matmul_bf16, linalg.to_bf16(a), linalg.to_bf16(b)
        for arm, env in ARMS.items():
            os.environ.update(env)
            out = fn(x, y)  # warm: pipelines, pool growth
            ts = []
            for _ in range(args.reps):
                t0 = time.perf_counter()
                out = fn(x, y)
                ts.append((time.perf_counter() - t0) * 1e3)
            sha = hashlib.sha256(np.asarray(out).tobytes()).hexdigest()[:16]
            print("GEMM-PROBE lane=%s n=%d arm=%s median_ms=%.2f min_ms=%.2f sha=%s"
                  % (lane, n, arm, statistics.median(ts), min(ts), sha), flush=True)


if __name__ == "__main__":
    main()
