#!/usr/bin/env python3
"""Capture offline CUDA resources; this does not run a GPU or measure occupancy.

Accepts retained .ptx[.gz]. Requires a compatible CUDA toolkit, but no GPU.
The assembler result is specific to that toolkit, not necessarily the runtime
driver's JIT result. See https://docs.nvidia.com/cuda/cuda-binary-utilities/.
"""

import argparse
import gzip
import hashlib
import json
import platform
import re
import shutil
import subprocess
from pathlib import Path


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def inspect_ptx(data):
    text = data.decode("utf-8")
    targets = re.findall(r"^\s*\.target\s+(sm_\w+)", text, re.M)
    entries = re.findall(r"\.entry\s+([^\s(]+)", text)
    if len(targets) != 1 or not entries:
        raise ValueError("Expected one PTX target and at least one kernel entry")
    return {"target": targets[0], "entries": entries, "ptx_sha256": sha256(data)}


def capture(source, output, ptxas, cuobjdump):
    packed = source.read_bytes()
    data = gzip.decompress(packed) if source.suffix == ".gz" else packed
    report = inspect_ptx(data)
    report.update({"source": str(source.resolve()), "source_sha256": sha256(packed),
                   "host": platform.platform(), "status": "incomplete",
                   "scope": "offline assembly; no execution, timing, or achieved occupancy",
                   "runtime_jit_equivalence": "not established", "commands": []})
    # Refuse to overwrite evidence, including failed earlier attempts.
    output.mkdir(parents=True, exist_ok=False)
    (output / "input.ptx").write_bytes(data)

    def run(command, log):
        try:
            result = subprocess.run(command, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, timeout=180)
        except subprocess.TimeoutExpired as error:
            (output / log).write_bytes(error.stdout or b"")
            report["commands"].append({"argv": command, "log": log,
                                       "returncode": None, "timeout_seconds": 180})
            raise
        (output / log).write_bytes(result.stdout)
        report["commands"].append({"argv": command, "log": log,
                                   "returncode": result.returncode})
        if result.returncode:
            raise RuntimeError(f"{command[0]} exited {result.returncode}; see {log}")

    try:
        resolved = {}
        for name, executable in (("ptxas", ptxas), ("cuobjdump", cuobjdump)):
            path = shutil.which(executable)
            if not path:
                raise RuntimeError(f"Missing {name}: {executable}")
            resolved[name] = str(Path(path).resolve())
            report[name] = {"path": resolved[name],
                            "sha256": sha256(Path(path).read_bytes())}
            run([resolved[name], "--version"], name + "-version.log")
        run([resolved["ptxas"], "--verbose", "--gpu-name", report["target"],
             str((output / "input.ptx").resolve()), "--output-file",
             str((output / "kernel.cubin").resolve())], "ptxas.log")
        if not (output / "kernel.cubin").is_file() or not (output / "kernel.cubin").stat().st_size:
            raise RuntimeError("Assembler did not produce a nonempty cubin")
        for flag, log in (("--dump-resource-usage", "resources.log"),
                          ("--dump-sass", "sass.log")):
            run([resolved["cuobjdump"], flag,
                 str((output / "kernel.cubin").resolve())], log)
        report["status"] = "assembled"
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        report["error"] = str(error)
        raise
    finally:
        report["artifacts"] = {p.name: sha256(p.read_bytes())
                               for p in sorted(output.iterdir()) if p.is_file()}
        (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ptx", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--ptxas", default="ptxas")
    parser.add_argument("--cuobjdump", default="cuobjdump")
    args = parser.parse_args()
    try:
        report = capture(args.ptx, args.output, args.ptxas, args.cuobjdump)
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        parser.exit(1, f"Resource capture failed: {error}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
