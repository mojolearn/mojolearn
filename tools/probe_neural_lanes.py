#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Neural board lanes, OUR GPU arm only, with a breakdown (lane/gap-neural-models).

For each lane: the board's own inputs (tools/bench_board_neural.py make_inputs,
shape full) and runner (build_runner(lane, "ours")), one warm call, --reps
timed calls clocked the way the board clocks them (call + sync), then one call
under the libraries' stage-timing envs (MOJOLEARN_TRANSFORMER_TIMING,
MOJOLEARN_MAMBA_TIMING) and one under cProfile
(top binding calls by cumulative time). GPU lanes only; run on the boxes.

    python tools/probe_neural_lanes.py --lanes transformer-forward,samba-forward [--reps 5] [--env K=V ...]
"""
import argparse
import cProfile
import io
import os
import pstats
import statistics
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "python"))

TIMING_ENVS = ("MOJOLEARN_TRANSFORMER_TIMING", "MOJOLEARN_MAMBA_TIMING")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lanes", required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--env", action="append", default=[], help="K=V set before the runner is built")
    ap.add_argument("--no-profile", action="store_true")
    args = ap.parse_args()
    for kv in args.env:
        k, v = kv.split("=", 1)
        os.environ[k] = v
    import numpy as np
    import bench_board_neural as N
    for lane in args.lanes.split(","):
        steps = args.reps + 4
        work = tempfile.mkdtemp(prefix="probe-neural-")
        path = os.path.join(work, "in.npz")
        N.make_inputs(lane, "full", steps, path)
        with np.load(path) as z:
            data = {k: z[k] for k in z.files}
        r = N.build_runner(lane, "ours", "full", data)
        r.call(); r.sync()
        ts = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            r.call(); r.sync()
            ts.append((time.perf_counter() - t0) * 1e3)
        dg = r.digest() if hasattr(r, "digest") else None
        loss = getattr(r, "losses", None)
        print("NEURAL-PROBE lane=%s env=%s median_ms=%.3f min_ms=%.3f ms=%s digest=%s loss=%s" % (
            lane, ",".join(args.env) or "-", statistics.median(ts), min(ts),
            ",".join("%.2f" % t for t in ts), dg, (loss[:3] if loss else None)), flush=True)
        for e in TIMING_ENVS:
            os.environ[e] = "1"
        sys.stdout.flush()
        r.call(); r.sync()
        sys.stdout.flush()
        for e in TIMING_ENVS:
            os.environ.pop(e, None)
        if not args.no_profile:
            pr = cProfile.Profile()
            pr.enable()
            r.call(); r.sync()
            pr.disable()
            s = io.StringIO()
            pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(25)
            for line in s.getvalue().splitlines():
                if line.strip():
                    print("PROFILE %s | %s" % (lane, line))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
