#!/usr/bin/env python3
"""Focused CPU compile controller; no product execution or GPU qualification."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from build_sizing import box_resources, cgroup_cpu_limit
from cpu_quota import INHERITED_CAPS


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", nargs="+", choices=("scalers_pls", "scalers", "pls", "manifold", "umap_init", "jl_rp", "prep", "pca_query"),
                        default=["scalers", "pls", "manifold", "umap_init", "jl_rp"])
    parser.add_argument("--out", type=Path, default=Path(os.environ.get("LEG_OUT", "/root/results-peer2")))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    compiler = ROOT / ".pixi/envs/default/bin/mojo"
    cores, memory = box_resources()
    if cores < 1:
        raise RuntimeError("No CPU allocation detected")
    env = os.environ.copy()
    for name in INHERITED_CAPS:
        env.pop(name, None)
    env["MOJOLEARN_COMPILE_JOBS"] = str(cores)
    env["MOJOLEARN_BUILD_JOBS"] = "1"
    record = {
        "source_commit": os.environ.get("MOJOLEARN_COMMIT"),
        "coverage": "compiled exported forwarding roots and their transitive native calls only",
        "runtime": "not_run", "vendor_identity": "not_run",
        "compiler": str(compiler), "compiler_sha256": sha256(compiler),
        "compiler_version": subprocess.check_output([str(compiler), "--version"], text=True).strip(),
        "target": "x86-64-v3/cpu/identical",
        "resources": {"affinity": sorted(os.sched_getaffinity(0)), "cgroup_cpu_limit": cgroup_cpu_limit(),
                      "allocated_cores": cores, "available_bytes": memory, "compiler_jobs": cores,
                      "concurrent_builds": 1, "saturation": "not_claimed"},
        "compiles": [],
    }
    # Retain exact source bytes, including every local Mojo dependency (a
    # conservative superset) and dependency pins. The CPU runner retains this
    # entire output directory, including binaries, on the Mac and in R2.
    sources = sorted(ROOT.rglob("*.mojo"))
    sources = [path for path in sources if ".pixi" not in path.relative_to(ROOT).parts]
    sources += [ROOT / "pixi.lock", ROOT / "pixi.toml", Path(__file__).resolve()]
    manifest = {str(path.relative_to(ROOT)): sha256(path) for path in sources if path.is_file()}
    (args.out / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with tarfile.open(args.out / "compile-sources.tgz", "w:gz") as archive:
        for name in manifest:
            archive.add(ROOT / name, arcname=name)
    record["source_archive_sha256"] = sha256(args.out / "compile-sources.tgz")
    record["compiler_packages"] = {
        str(path.relative_to(compiler.parent.parent)): sha256(path)
        for path in sorted((compiler.parent.parent / "lib/mojo").rglob("*.mojoc"))
    }
    deadline = time.monotonic() + 2700  # Retention and teardown precede the 60-minute lease.
    for group in args.groups:
        source = Path("experiments/peer2_native_boundary_20261027") / f"compile_{group}.mojo"
        if group == "jl_rp":
            source = Path("experiments/peer2_decomposition_20261027/jl_rp_compile.mojo")
        elif group == "pca_query":
            source = Path("experiments/peer2_decomposition_20261027/pca_query_compile.mojo")
        elif group == "manifold":
            # Typed matrix/device owners also require their module registration
            # to compile; the real dedicated host entry covers that boundary.
            source = Path("bindings/entry/entry__mojolearn_runtime_manifold_host.mojo")
        binary = args.out / f"{group}.so"
        # Use the repository's supported activated toolchain, as build_host_family.sh
        # does. Invoking the executable alone omits the installed std search path.
        command = ["pixi", "run", "-e", "default", str(compiler), "build", "-j", str(cores), "--emit", "shared-lib",
                   "--target-cpu", "x86-64-v3", "-D", "MOJOLEARN_NUMERIC_IDENTICAL=1",
                   "-D", "MOJOLEARN_COLUMN_CPU", "-I", ".", "-I", "bindings",
                   str(source), "-o", str(binary)]
        started = time.monotonic()
        with (args.out / f"{group}.log").open("w") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                status = process.wait(timeout=max(1, min(1200, deadline - time.monotonic())))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                status = 124
        (args.out / f"{group}.exit").write_text(str(status) + "\n")
        record["compiles"].append({"group": group, "command": command, "exit_status": status,
                                   "seconds": time.monotonic() - started,
                                   "binary_sha256": sha256(binary) if binary.is_file() else None})
        (args.out / "compile-summary.json").write_text(json.dumps(record, indent=2) + "\n")
        print(f"{group}: exit={status}", flush=True)
    return int(any(item["exit_status"] != 0 for item in record["compiles"]))


if __name__ == "__main__":
    raise SystemExit(main())
