#!/usr/bin/env python3
"""External fault-control runner with process-group timeouts and retained logs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command, log, budget):
    start = time.monotonic()
    expired = False
    with log.open("w") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=budget)
        except subprocess.TimeoutExpired:
            expired = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                code = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait(timeout=5)
    record = {"command": command, "exit_code": code, "timed_out": expired,
              "budget_seconds": budget, "elapsed_seconds": time.monotonic() - start, "log": str(log)}
    log.with_suffix(".status.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qualified-out", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    check = Path(__file__).with_name("check_native_adapters.py").resolve()
    native = args.qualified_out / "libmojolearn_adapter.so"
    extension = args.qualified_out / "mojolearn_torch.so"
    original = {"native": digest(native), "extension": digest(extension)}
    baseline = json.loads((args.qualified_out / "adapter-results.json").read_text())
    assert baseline["complete"] and baseline.get("fault", "none") == "none"
    report = {"complete": False, "original": original, "controls": [], "failures": []}
    for fault in ("wrong-operation", "stall"):
        directory = args.out / fault
        directory.mkdir()
        shutil.copy2(native, directory / native.name)
        command = [sys.executable, str(check), "--out", str(directory), "--device", args.device, "--fault", fault]
        build = run(command + ["--stage", "build"], directory / "build.log", 300)
        if build["exit_code"] or build["timed_out"]:
            report["failures"].append({"fault": fault, "stage": "build", "result": build})
            continue
        outcome = run(command + ["--stage", "test"], directory / "test.log", 12 if fault == "stall" else 120)
        if fault == "stall":
            caught = outcome["timed_out"] and outcome["exit_code"] != 0
        else:
            results_path = directory / "adapter-results.json"
            results = json.loads(results_path.read_text()) if results_path.exists() else {}
            failed = {item["name"] for item in results.get("failures", [])}
            caught = outcome["exit_code"] == 1 and not outcome["timed_out"] and "add" in failed
        report["controls"].append({"fault": fault, "detected": caught, "result": outcome})
        if not caught:
            report["failures"].append({"fault": fault, "stage": "detection"})
    # Reuse the original accepted binaries; a restored run must pass after
    # both fault subprocesses have ended, including the deliberately stalled one.
    restored = run([sys.executable, str(check), "--out", str(args.qualified_out), "--device", args.device,
                    "--stage", "test"], args.out / "restored.log", 120)
    report["restored"] = restored
    report["original_unchanged"] = original == {"native": digest(native), "extension": digest(extension)}
    if restored["exit_code"] or restored["timed_out"] or not report["original_unchanged"]:
        report["failures"].append({"stage": "restore"})
    report["complete"] = not report["failures"] and len(report["controls"]) == 2
    (args.out / "fault-results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"complete": report["complete"], "controls": [
        {"fault": x["fault"], "detected": x["detected"]} for x in report["controls"]],
        "restored_exit": restored["exit_code"]}))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
