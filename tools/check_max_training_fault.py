#!/usr/bin/env python3
"""Compile a deliberately wrong MAX training adapter, prove rejection, restore."""
import argparse
import json
from pathlib import Path
import shutil
import sys
from check_native_adapter_faults import digest, run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--native-library", type=Path, required=True)
    p.add_argument("--torch-results", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    source = a.out / "max_ops_wrong_lr"
    shutil.copytree(root / "adapters/max_ops", source)
    module = source / "__init__.mojo"
    text = module.read_text()
    original = 'var status = lib.get_function[Int]("mojolearn_adapter_train")('
    assert text.count(original) == 1
    module.write_text(text.replace(original, 'floats[0] *= Float64(2)\n    ' + original))
    original_hashes = [digest(a.library), digest(a.native_library)]
    fault_package = a.out / "mojolearn_wrong_training.mojoc"
    build = run(["pixi", "run", "mojo", "package", str(source), "-o", str(fault_package)], a.out / "build.log", 300)
    report = {"complete": False, "build": build, "original": original_hashes}
    def command(package, output):
        return [sys.executable, str(root / "tools/check_max_training_adapters.py"), "--library", str(package),
                "--native-library", str(a.native_library), "--torch-results", str(a.torch_results),
                "--out", str(output), "--smoke"]
    if not build["exit_code"] and not build["timed_out"]:
        outcome = run(command(fault_package, a.out / "wrong"), a.out / "wrong.log", 180)
        result_path = a.out / "wrong/max-training-results.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        caught = outcome["exit_code"] == 1 and not outcome["timed_out"] and any("AssertionError" in f for f in result.get("failures", []))
        restored = run(command(a.library, a.out / "restored"), a.out / "restored.log", 180)
        unchanged = original_hashes == [digest(a.library), digest(a.native_library)]
        report.update(outcome=outcome, detected=caught, restored=restored, original_unchanged=unchanged,
                      complete=caught and unchanged and restored["exit_code"] == 0 and not restored["timed_out"])
    (a.out / "max-training-fault-results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"complete": report["complete"], "detected": report.get("detected")}))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
