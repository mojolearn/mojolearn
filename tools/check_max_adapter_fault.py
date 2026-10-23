#!/usr/bin/env python3
"""External MAX wrong-dispatch control; preserve the accepted package unchanged."""
import argparse
import hashlib
import json
from pathlib import Path
from check_native_adapter_faults import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.out.mkdir(parents=True)
    source_dir = args.out / "source"
    source_dir.mkdir()
    original = (root / "adapters/max_ops/__init__.mojo").read_text()
    target = "execute_operation[target, 6, library]"
    assert original.count(target) == 1
    (source_dir / "__init__.mojo").write_text(original.replace(target, "execute_operation[target, 7, library]"))
    package = args.out / "wrong-relu.mojoc"
    original_hash = hashlib.sha256(args.library.read_bytes()).hexdigest()
    build = run(["pixi", "run", "mojo", "precompile", str(source_dir), "-o", str(package)],
                args.out / "build.log", 120)
    checker = ["pixi", "run", "python", str(root / "tools/check_max_native_adapters.py"),
               "--native-library", str(args.native_library)]
    detected = False
    if build["exit_code"] == 0 and not build["timed_out"]:
        fault_dir = args.out / "fault"
        outcome = run(checker + ["--library", str(package), "--out", str(fault_dir)], args.out / "fault.log", 180)
        results_path = fault_dir / "max-results.json"
        results = json.loads(results_path.read_text()) if results_path.exists() else {}
        failures = results.get("failures", [])
        detected = outcome["exit_code"] == 1 and not outcome["timed_out"] and any("AssertionError" in x for x in failures)
    restored = run(checker + ["--library", str(args.library), "--out", str(args.out / "restored")],
                   args.out / "restored.log", 300)
    unchanged = original_hash == hashlib.sha256(args.library.read_bytes()).hexdigest()
    complete = detected and unchanged and restored["exit_code"] == 0 and not restored["timed_out"]
    report = {"complete": complete, "fault": "relu dispatched as sigmoid", "detected": detected,
              "original_package_unchanged": unchanged, "restored": restored, "build": build}
    (args.out / "max-fault-results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"complete": complete, "detected": detected, "restored_exit": restored["exit_code"]}))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
