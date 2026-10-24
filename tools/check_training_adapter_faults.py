#!/usr/bin/env python3
"""External wrong-learning-rate and stall controls; retain faults separately."""
import argparse
import json
from pathlib import Path
import shutil
import sys
from check_native_adapter_faults import digest, run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--qualified-out", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--host-staging", action="store_true")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parent
    native, extension = [a.qualified_out / n for n in ("libmojolearn_adapter.so", "mojolearn_torch.so")]
    original = [digest(native), digest(extension)]
    report = {"complete": False, "controls": [], "failures": [], "original": original}
    for fault in ("training-wrong-lr", "training-stall"):
        directory = a.out / fault
        directory.mkdir()
        shutil.copy2(native, directory / native.name)
        build = [sys.executable, str(root / "check_native_adapters.py"), "--out", str(directory),
                 "--stage", "build", "--fault", fault]
        if a.host_staging:
            build += ["--host-staging"]
        built = run(build, directory / "build.log", 300)
        if built["exit_code"] or built["timed_out"]:
            report["failures"].append({"fault": fault, "stage": "build"})
            continue
        test = [sys.executable, str(root / "check_native_training_adapters.py"), "--extension",
                str(directory / extension.name), "--out", str(directory), "--device", a.device, "--smoke"]
        outcome = run(test, directory / "test.log", 12 if fault.endswith("stall") else 120)
        if fault.endswith("stall"):
            caught = outcome["timed_out"] and "TRAINING_FAULT_STALL_ENTERED" in (directory / "test.log").read_text()
        else:
            path = directory / "training-results.json"
            result = json.loads(path.read_text()) if path.exists() else {}
            caught = outcome["exit_code"] == 1 and not outcome["timed_out"] and any(
                "AssertionError" in f and "analytic" in f for f in result.get("failures", []))
        report["controls"].append({"fault": fault, "detected": caught, "outcome": outcome})
        if not caught:
            report["failures"].append({"fault": fault, "stage": "detection"})
    restored = run([sys.executable, str(root / "check_native_training_adapters.py"), "--extension", str(extension),
                    "--out", str(a.out / "restored"), "--device", a.device, "--smoke"], a.out / "restored.log", 120)
    report["restored"] = restored
    report["original_unchanged"] = original == [digest(native), digest(extension)]
    if restored["exit_code"] or restored["timed_out"] or not report["original_unchanged"]:
        report["failures"].append({"stage": "restore"})
    report["complete"] = len(report["controls"]) == 2 and not report["failures"]
    (a.out / "training-fault-results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"complete": report["complete"], "controls": report["controls"]}))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
