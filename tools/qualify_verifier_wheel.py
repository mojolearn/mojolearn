#!/usr/bin/env python3
"""Qualify the installed verifier CLI on one exact wheel, outside the checkout."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile
import zipfile


def admit(kind, doc, models):
    if kind == "coverage":
        assert doc["lanes"] and doc["cpu_execution_counts"]["declared"] > 0
    elif kind == "self-test":
        assert doc["passed"] is True
        assert doc["clean"]["state"] == "IDENTICAL"
        assert doc["perturbed"]["state"] == "DIVERGENT"
    else:
        cells = {(r["lane"], r["fixture"], r["part"]): r for r in doc["cells"]}
        assert len(cells) == len(doc["cells"]), "duplicate cell"
        if kind == "extended":
            owed = [r for r in cells.values() if r["state"] == "OWED"]
            assert (doc["exit"], doc["verdict"]) == ((5, "INCOMPLETE") if owed else (0, "VERIFIED"))
            for row in cells.values():
                assert row["state"] in ("IDENTICAL", "N/A", "OWED")
                if row["state"] == "OWED":
                    assert row["part"] in ("batchgrad", "batchscale", "ragged", "rlpair")
                    assert row.get("error") is None and re.fullmatch("[0-9a-f]{16}", row["value"])
        else:
            assert doc["exit"] == 0 and doc["verdict"] == "VERIFIED"
        if kind == "models":
            expected = {(f"portable:{m['lane']}", m["fixture"], part)
                        for m in models for part in ("model", "batch")}
            assert expected and cells.keys() == expected
            assert all(row["state"] == "IDENTICAL" for row in cells.values())
            assert doc["selection"]["models_only"] is True
        else:
            for part in ("train", "infer", "batch"):
                assert cells["knn", "base", part]["state"] == "IDENTICAL"
            if kind == "extended":
                assert {"batchgrad", "batchscale", "ragged", "rlpair"} <= {
                    row["part"] for row in cells.values()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default="python3.12")
    parser.add_argument("--wheelhouse", type=Path)
    args = parser.parse_args()
    wheel, output = args.wheel.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    receipt = output / "results.json"
    if receipt.exists():
        parser.error("use a fresh output directory")
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    with zipfile.ZipFile(wheel) as archive:
        models = json.loads(archive.read("mojolearn/verify_reference/models/models.json"))["models"]
    manifest = dict(wheel=str(wheel), wheel_sha256=digest, status="INCOMPLETE", jobs=[],
                    scope="installed CLI, portable CPU models, one GPU lane with batch checks, self-test")
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("MOJOLEARN_") and k not in ("PYTHONPATH", "PYTHONHOME")}
    env.update(MOJOLEARN_NUMERIC_MODE="identical", PYTHONNOUSERSITE="1")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MOJOLEARN_CPU_THREADS"):
        env[name] = "1"

    def save():
        receipt.write_text(json.dumps(manifest, indent=2) + "\n")

    def run(name, command, cwd, json_output=False, allowed_exits=(0,)):
        path = output / (name + (".json" if json_output else ".log"))
        with path.open("w") as stdout, (output / (name + ".stderr.log")).open("w") as stderr:
            proc = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                    stdout=stdout, stderr=stderr, start_new_session=True)
            try:
                code = proc.wait(timeout=180)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                code = 124
        manifest["jobs"].append(dict(name=name, exit_code=code, output=str(path)))
        save()
        if code not in allowed_exits:
            raise RuntimeError(f"{name} failed with exit {code}; see {path}")
        return json.loads(path.read_text()) if json_output else None

    save()
    try:
        with tempfile.TemporaryDirectory(prefix="mojolearn-verifier-wheel-") as directory:
            work = Path(directory)
            run("create-venv", [args.python, "-m", "venv", str(work / "venv")], work)
            python = str(work / "venv/bin/python")
            offline = ["--no-index", "--find-links", str(args.wheelhouse.resolve())] if args.wheelhouse else []
            run("install", [python, "-m", "pip", "install", "--no-input", "--only-binary=:all:",
                            *offline, str(wheel), "numpy>=1.24"], work)
            run("dependencies", [python, "-m", "pip", "check"], work)
            guard = """import json,pathlib,sys,mojolearn
p=pathlib.Path(mojolearn.__file__).resolve()
assert p.is_relative_to(pathlib.Path(sys.prefix).resolve()) and 'site-packages' in p.parts
assert mojolearn.__version__ == sys.argv[1]
print(json.dumps(dict(package=str(p),version=mojolearn.__version__,vendor=mojolearn.vendor())))
"""
            manifest["installed"] = run("installed", [python, "-c", guard, wheel.name.split("-")[1]], work, True)
            commands = {
                "coverage": ["--coverage"],
                "models": ["--models-only", "--repeats", "2"],
                "batch": ["--lanes", "knn", "--fixtures", "base", "--repeats", "2", "--no-models"],
                "extended": ["--lanes", "knn", "--fixtures", "base", "--repeats", "2", "--batch-checks", "--no-models"],
                "self-test": ["--self-test"],
            }
            for kind, flags in commands.items():
                doc = run(kind, [python, "-m", "mojolearn", "verify", *flags, "--json"], work, True,
                          (0, 5) if kind == "extended" else (0,))
                admit(kind, doc, models)
                if kind == "extended":
                    # This gate checks the CLI's honest incomplete result;
                    # it does not qualify properties lacking references.
                    manifest["unqualified_properties"] = [
                        dict(lane=r["lane"], fixture=r["fixture"], part=r["part"], state=r["state"])
                        for r in doc["cells"] if r["state"] == "OWED"]
            assert hashlib.sha256(wheel.read_bytes()).hexdigest() == digest, "wheel changed"
            manifest["status"] = "PASSED"
    except Exception as exc:
        manifest.update(status="FAILED", reason=repr(exc))
    save()
    print(json.dumps(manifest, indent=2))
    return 0 if manifest["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
