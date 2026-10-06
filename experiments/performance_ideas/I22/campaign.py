#!/usr/bin/env python3
"""Compile/validate independent TSQR schedules; never time the correctness fixture."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys

ARMS = {
    "incumbent": [],
    "reuse": ["MOJOLEARN_IDN_TSQR_REUSE=1"],
    "strip": ["MOJOLEARN_IDN_TSQR_STRIP_UPDATE=1"],
    "combined": ["MOJOLEARN_IDN_TSQR_REUSE=1", "MOJOLEARN_IDN_TSQR_STRIP_UPDATE=1"],
    "legacy": ["MOJOLEARN_IDN_TSQR_GRID_OFF=1", "MOJOLEARN_IDN_TSQR_NORM_OFF=1"],
}

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("render", "build", "validate"), required=True)
    parser.add_argument("--vendor", choices=("nvidia", "amd", "apple"), required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--mojo", default="mojo")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=(*ARMS, "all"), default="all")
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo, text=True).strip()
    if head != args.source_sha or dirty:
        parser.error("campaign requires the requested clean frozen source")
    args.output.mkdir(parents=True, exist_ok=True)
    result = 0
    for arm in ARMS if args.arm == "all" else (args.arm,):
        binary = args.output / f"I22-{args.vendor}-{arm}"
        build = [sys.executable, str(repo / "gemm/experiments/native_build.py"),
                 str(repo / "experiments/performance_ideas/I22/check.mojo"),
                 "--vendor", args.vendor, "--mode", "identical", "--source-sha", head,
                 "--mojo", args.mojo, "--output", str(binary)]
        for define in ARMS[arm]:
            build.extend(("--define", define))
        command = build if args.stage in ("render", "build") else [str(binary)]
        if args.stage == "render":
            print(json.dumps({"arm": arm, "build_argv": build, "validation_argv": [str(binary)],
                              "source_sha": head, "defines": ARMS[arm]}))
            continue
        log = args.output / f"I22-{args.vendor}-{arm}-{args.stage}.log"
        with log.open("w") as stream:
            try:
                completed = subprocess.run(command, cwd=repo, stdout=stream, stderr=subprocess.STDOUT,
                                           timeout=args.timeout, check=False)
                code = completed.returncode
            except subprocess.TimeoutExpired:
                code = 124
            except OSError as exc:
                stream.write(str(exc) + "\n")
                code = 127
        after = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        dirty_after = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo, text=True).strip()
        frozen = after == head and not dirty_after
        receipt = {"id": "I22", "arm": arm, "stage": args.stage, "vendor": args.vendor,
                   "mode": "identical", "defines": ARMS[arm], "source_sha": head,
                   "source_frozen": frozen, "argv": command, "exit_code": code,
                   "qualification": "compile_only" if args.stage == "build" else "local_fixture_only",
                   "remaining_gates": ["same-version cross-vendor identity", "full-operation NVIDIA/AMD timing"],
                   "log": str(log)}
        (args.output / f"I22-{args.vendor}-{arm}-{args.stage}.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"I22 {arm} {args.stage} exit_code={code} source_frozen={frozen}")
        if code or not frozen:
            result = 1
    return result

if __name__ == "__main__":
    raise SystemExit(main())
