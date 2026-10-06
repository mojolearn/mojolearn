#!/usr/bin/env python3
"""Adapt the real full forest/GBDT driver to the full A/B queue JSON contract.

No build, tiny fixture, opponent run, quality admission or default promotion.
This adapter was written but not executed in the source-only task.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
FIT_BOUNDARY = "caller buffer preparation, public constructor+fit+synchronization, held-out scoring and consumption of all requested outputs; fit and inference timings also retained separately"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--infer", action="store_true")
    args = parser.parse_args()
    if os.environ.get("MOJOLEARN_AFT_CAPTURE") != "1" or os.environ.get("MOJOLEARN_BENCH_INSTALLED") != "1":
        raise ValueError("use the isolated Apple FAST tree pipeline worker")
    argv = [sys.executable, str(ROOT / "bench/speed/forest_speed_arm.py"), "--lane", args.lane,
            "--dataset", args.dataset, "--ours-only"]
    if args.infer:
        # This is an unbounded harness selector, not a kernel size threshold:
        # the existing driver takes min(requested rows, full training rows).
        argv += ["--infer", "--infer-large-rows", str(sys.maxsize)]
    log = args.output.with_suffix(".forest.log")
    with log.open("x") as stream:
        rc = subprocess.run(argv, env=os.environ.copy(), stdout=stream, stderr=subprocess.STDOUT).returncode
    fits, inference, failures, headers, operations = [], [], [], [], []
    metadata = None
    with log.open() as stream:
        for line in stream:
            if line.startswith("FSPEED-AFT "):
                metadata = json.loads(line[len("FSPEED-AFT "):])
            elif line.startswith("FSPEED-OPERATION "):
                operation = json.loads(line[len("FSPEED-OPERATION "):])
                if operation.get("arm") == "ours":
                    operations.append(operation)
            elif line.startswith(("FSPEED ", "FSPEED-INFER ", "FSPEED-HEADER ")):
                record = dict(token.split("=", 1) for token in line.split()[1:] if "=" in token)
                if record.get("arm") != "ours":
                    continue
                if line.startswith("FSPEED "):
                    fits.append(record)
                elif line.startswith("FSPEED-INFER "):
                    inference.append(record)
                else:
                    headers.append(record)
            elif line.startswith("FSPEED-REFUSED "):
                failures.append(line.rstrip())
    if rc or failures or metadata is None or len(fits) != 1:
        raise RuntimeError(f"full workload incomplete rc={rc}, fits={len(fits)}, refusals={len(failures)}; {log}")
    if len(operations) != 1 or not operations[0].get("consumed_output_hashes"):
        raise RuntimeError("whole-operation completion/consumption evidence is missing")
    if not headers or any(h.get("mode") != "FAST" or h.get("size") != "shipped" for h in headers):
        raise RuntimeError("driver mode/size did not match Apple FAST full recipe")
    if metadata["mode"] != "fast" or metadata["vendor"] != "apple":
        raise RuntimeError("driver mode/vendor changed")
    if args.infer and not inference:
        raise RuntimeError("required inference phase is missing")
    state = metadata["states"].get("ours", {})
    if not state.get("output_sha256") or not metadata["metrics"].get("ours"):
        raise RuntimeError("consumed output hashes or held-out metrics are missing")
    result = dict(metadata, contract="aft-full-workload-v1", status="CAPTURED", lane=args.lane,
                  requested_dataset=args.dataset, timed_boundary=FIT_BOUNDARY,
                  output_sha256=state["output_sha256"], timings={"whole_operation": operations, "fit": fits, "inference": inference},
                  internal_warmup="none when AFT_EXTERNAL_WARMUP=1; outer queue supplies one excluded warmup capture per arm in a separate process",
                  quality_status="pending", promotion_authorized=False, full_log=str(log))
    if state.get("model", {}).get("status") == "ok":
        result["model_sha256"] = state["model"]["sha256"]
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
