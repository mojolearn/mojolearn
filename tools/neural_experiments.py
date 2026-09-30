#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""One build, many toggles: run tools/neural_stage_timing.py under each
experiment's environment and print one table.

lane/neural-net-experiment (2026-09-30). Every experiment on the branch is
a runtime toggle, so one wheel serves every A/B.
Each configuration runs in its OWN subprocess (the bindings read most
toggles at load or at first use), for the lanes given, `--calls` calls
each; the table shows the median after the first call, the ratio to the
baseline, and whether the output DIGEST equals the baseline's (`same` /
`MOVED` / `n/a` on training lanes, whose LOSSES are compared instead).

    python tools/neural_experiments.py                       # the default set, six lanes
    python tools/neural_experiments.py --lane transformer-forward --lane samba-forward
    python tools/neural_experiments.py --set nvidia          # the NVIDIA-shaped set
    python tools/neural_experiments.py --set amd
    python tools/neural_experiments.py --gemm-arms shipped,tuned128,half,quarter,kpack
    python tools/neural_experiments.py --only speculative_attn,swiglu_fused
    python tools/neural_experiments.py --json results.json

A configuration whose digest MOVED is not a speed result; it is a bug
report against that toggle (or a stage the toggle legitimately drops from
the card), and it must not be kept.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TIMING = HERE / "neural_stage_timing.py"
LANES = ("transformer-forward", "mamba3-forward", "samba-forward",
         "samba-train-step", "lm-forward", "lm-train-step")

# name -> environment delta over the baseline. Baseline = the branch's
# defaults (retention on, stage reset on, per-layer sync on, fused SwiGLU
# off, speculative attention off, session fresh path on, shipped GEMM arm).
EXPERIMENTS = {
    "baseline": {},
    "no_retain_weights": {"MOJOLEARN_TRANSFORMER_RETAIN_WEIGHTS": "0",
                          "MOJOLEARN_MAMBA3_RETAIN_WEIGHTS": "0"},
    "legacy_fresh_entry": {"MOJOLEARN_TRANSFORMER_SESSION_FRESH": "0"},
    "legacy_everything": {"MOJOLEARN_TRANSFORMER_LEGACY_SETUP": "1",
                          "MOJOLEARN_MAMBA3_LEGACY_SETUP": "1"},
    "no_stage_reset": {"MOJOLEARN_TRANSFORMER_STAGE_RESET": "0"},
    "speculative_attn": {"MOJOLEARN_ATTN_SPECULATIVE": "1"},
    "swiglu_fused": {"MOJOLEARN_SWIGLU_FUSED": "1"},
    "no_layer_sync": {"MOJOLEARN_BYTE_LM_LAYER_SYNC": "0"},
    "mamba3_legacy": {"MOJOLEARN_MAMBA3_LEGACY_SETUP": "1"},
    "mamba3_no_retain_stages": {"MOJOLEARN_MAMBA3_RETAIN_STAGES": "0"},
    "norm_dw_own_ws": {"MOJOLEARN_TRANSFORMER_NORM_DW_OWN_WS": "1"},
    # the S16 q/k backward arms (lane/neural-net-experiment, the S16 pass):
    # the default is `regs2`; each arm is the same chains in the same order
    "s16_naive": {"MOJOLEARN_MAMBA3_S16_QK_ARM": "naive"},
    "s16_shared": {"MOJOLEARN_MAMBA3_S16_QK_ARM": "shared"},
    "s16_regs": {"MOJOLEARN_MAMBA3_S16_QK_ARM": "regs"},
    "s16_smem48": {"MOJOLEARN_MAMBA3_S16_QK_ARM": "smem48"},
    "all_on": {"MOJOLEARN_ATTN_SPECULATIVE": "1", "MOJOLEARN_SWIGLU_FUSED": "1",
               "MOJOLEARN_BYTE_LM_LAYER_SYNC": "0", "MOJOLEARN_TRANSFORMER_STAGE_RESET": "0"},
}
SETS = {
    "priority": ["baseline", "mamba3_legacy", "mamba3_no_retain_stages", "norm_dw_own_ws"],
    # the S16 arms on the Mamba lanes (run with --lane mamba3-forward --lane
    # samba-train-step, or the timing tool for the per-kernel walls)
    "s16": ["baseline", "s16_naive", "s16_shared", "s16_regs", "s16_smem48"],
    "default": ["baseline", "no_retain_weights", "no_stage_reset", "speculative_attn",
                "swiglu_fused", "no_layer_sync", "all_on"],
    "nvidia": ["baseline", "no_retain_weights", "legacy_fresh_entry", "no_stage_reset",
               "speculative_attn", "swiglu_fused", "no_layer_sync", "all_on"],
    "amd": ["baseline", "legacy_everything", "no_retain_weights", "no_stage_reset",
            "speculative_attn", "swiglu_fused", "no_layer_sync", "all_on"],
}


def run_one(name, env_delta, lanes, calls, shape):
    env = dict(os.environ)
    env.update(env_delta)
    cmd = [sys.executable, str(TIMING), "--calls", str(calls), "--shape", shape]
    for lane in lanes:
        cmd += ["--lane", lane]
    print("### %s  %s" % (name, " ".join("%s=%s" % kv for kv in sorted(env_delta.items())) or "(defaults)"),
          flush=True)
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    res = {"name": name, "env": env_delta, "returncode": proc.returncode,
           "median": {}, "first": {}, "digest": {}, "losses": {}, "stderr_tail": proc.stderr[-2000:]}
    for line in proc.stdout.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "SUMMARY" and len(parts) >= 4:
            lane = parts[1]
            for tok in parts[2:]:
                if tok.startswith("first="):
                    res["first"][lane] = float(tok[6:])
                elif tok.startswith("median_after_first="):
                    res["median"][lane] = float(tok[len("median_after_first="):])
        elif parts[0] == "DIGEST" and len(parts) >= 3:
            res["digest"][parts[1]] = " ".join(parts[2:])
        elif parts[0] == "LOSSES" and len(parts) >= 3:
            res["losses"][parts[1]] = " ".join(parts[2:])
    if proc.returncode != 0:
        print("    FAILED (exit %d); stderr tail:\n%s" % (proc.returncode, proc.stderr[-1500:]), flush=True)
    else:
        for lane in lanes:
            print("    %-20s %8.3f ms  digest=%s" % (lane, res["median"].get(lane, float("nan")),
                                                     res["digest"].get(lane, "?")), flush=True)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lane", action="append", choices=LANES)
    ap.add_argument("--shape", default="full")
    ap.add_argument("--calls", type=int, default=6)
    ap.add_argument("--set", default="default", choices=sorted(SETS))
    ap.add_argument("--only", help="comma-separated experiment names (baseline is always run)")
    ap.add_argument("--gemm-arms", help="comma-separated MOJOLEARN_GEMM_ARM names to add as experiments")
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    lanes = args.lane or list(LANES)
    names = SETS[args.set]
    if args.only:
        names = ["baseline"] + [n for n in args.only.split(",") if n and n != "baseline"]
    exps = {n: EXPERIMENTS[n] for n in names}
    if args.gemm_arms:
        for arm in args.gemm_arms.split(","):
            arm = arm.strip()
            if arm and arm != "shipped":
                exps["gemm_arm_" + arm] = {"MOJOLEARN_GEMM_ARM": arm}
    results = []
    for name, delta in exps.items():
        results.append(run_one(name, delta, lanes, args.calls, args.shape))
    base = results[0]
    print("\n| experiment | " + " | ".join(lanes) + " |")
    print("|---|" + "---:|" * len(lanes))
    for r in results:
        cells = []
        for lane in lanes:
            m = r["median"].get(lane)
            b = base["median"].get(lane)
            if m is None or r["returncode"] != 0:
                cells.append("FAILED")
                continue
            ratio = (" (%.2fx)" % (m / b)) if b else ""
            if r is base:
                bits = ""
            elif r["digest"].get(lane, "None") not in ("None", "?") and base["digest"].get(lane) not in (None, "None"):
                bits = " same" if r["digest"][lane] == base["digest"][lane] else " **MOVED**"
            elif lane in r["losses"] and lane in base["losses"]:
                bits = " same" if r["losses"][lane] == base["losses"][lane] else " **MOVED**"
            else:
                bits = " n/a"
            cells.append("%.3f%s%s" % (m, ratio, bits))
        print("| %s | %s |" % (r["name"], " | ".join(cells)))
    print("\nratio < 1 is faster than baseline; MOVED means the output bits differ from baseline: do not keep that toggle.")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
