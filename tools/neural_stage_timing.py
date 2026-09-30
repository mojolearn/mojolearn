#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Per-stage timing of the board's neural cells, on the box in front of you.

lane/neural-net-experiment (2026-09-30). The board records one number per
cell; this prints where it goes. It builds the SAME inputs and the SAME
`ours` runners `tools/bench_board_neural.py` races (same seed, same shape
table), sets the bindings' stage-tick environment BEFORE they load, calls
each cell `--calls` times and prints every call's wall time next to the
ticks the bindings print (`timing surface.* ms`, `step.*`, `block.*`,
`attn.*`, `M3_PHASE ...`). For the LM lanes it also prints the resident
session's `attention_stage_report()` afterwards: which layers grew the
quadratic attention stages, which is the 3.7 GB training-step question.

    python tools/neural_stage_timing.py                      # all six lanes, 5 calls
    python tools/neural_stage_timing.py --lane lm-train-step --calls 20
    MOJOLEARN_GEMM_ARM=tuned128 python tools/neural_stage_timing.py --lane lm-train-step
    MOJOLEARN_BYTE_LM_LAYER_SYNC=0 python tools/neural_stage_timing.py --lane lm-train-step

Read the first call as "session open + compile" and the rest as the
steady state; the board's warm-up round is the first call.
"""
import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MOJOLEARN_TRANSFORMER_TIMING", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

LANES = ("transformer-forward", "mamba3-forward", "samba-forward",
         "samba-train-step", "lm-forward", "lm-train-step")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lane", action="append", choices=LANES,
                    help="repeatable; default: all six")
    ap.add_argument("--shape", default="full", help="bench_board_neural shape name (default full)")
    ap.add_argument("--calls", type=int, default=5)
    ap.add_argument("--json", help="also write {lane: [ms, ...]} here")
    args = ap.parse_args(argv)
    import bench_board_neural as bbn
    lanes = args.lane or list(LANES)
    out = {}
    with tempfile.TemporaryDirectory() as work:
        for lane in lanes:
            path = os.path.join(work, lane + ".npz")
            bbn.make_inputs(lane, args.shape, args.calls + 1, path)
            import numpy as np
            with np.load(path) as z:
                data = {k: z[k] for k in z.files}
            print("=== %s shape=%s" % (lane, args.shape), flush=True)
            runner = bbn.build_runner(lane, "ours", args.shape, data)
            ms = []
            for k in range(args.calls):
                t0 = time.perf_counter()
                runner.call()
                runner.sync()
                dt = (time.perf_counter() - t0) * 1000.0
                ms.append(dt)
                print("CALL %d %s %.3f ms" % (k, lane, dt), flush=True)
            out[lane] = ms
            steady = sorted(ms[1:]) if len(ms) > 1 else ms
            print("SUMMARY %s first=%.3f median_after_first=%.3f ms" % (
                lane, ms[0], steady[len(steady) // 2]), flush=True)
            trainer = getattr(runner, "trainer", None)
            report = getattr(trainer, "attention_stage_report", None)
            if callable(report):
                try:
                    print("ATTENTION_STAGE_REPORT %s %s" % (
                        lane, json.dumps(report(), sort_keys=True, default=str)), flush=True)
                except Exception as exc:  # the report is diagnostic; never fail the run
                    print("ATTENTION_STAGE_REPORT %s unavailable (%s)" % (lane, exc), flush=True)
            info = getattr(runner, "info", None)
            if info:
                print("INFO %s %s" % (lane, json.dumps(info, sort_keys=True, default=str)), flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
