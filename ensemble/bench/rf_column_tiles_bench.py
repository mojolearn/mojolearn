#!/usr/bin/env python3
"""Opt-in column-tile ABBA full-fit experiment; holds shared build lock.

Run from repository root through tools/with_build_lock.sh. No default change.
"""
import argparse, hashlib, json, os, pathlib, statistics, subprocess
ROOT = pathlib.Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["fast", "identical"], default="fast")
p.add_argument("--rows", type=int, default=65537)
p.add_argument("--cols", type=int, default=28)
p.add_argument("--build-only", action="store_true")
p.add_argument("--skip-build", action="store_true")
p.add_argument("--output-dir", type=pathlib.Path, default=ROOT / "bench/results/rf_column_tiles_2026-09-09")
p.add_argument("--binary-dir", type=pathlib.Path)
p.add_argument("--prewarm", action="store_true", help="discard one complete five-fit invocation per binary before measuring")
a = p.parse_args()
assert os.environ.get("MOJOLEARN_BUILD_LOCK_HELD") == "1", "hold tools/with_build_lock.sh for the entire run"
out = a.output_dir.resolve()
bin_dir = (a.binary_dir or out).resolve()
bin_dir.mkdir(parents=True, exist_ok=True)
out.mkdir(parents=True, exist_ok=True)
variants = ["reference", "columns2", "columns4"]
for variant in variants:
    binary = bin_dir / (a.mode + "_" + variant)
    if a.skip_build:
        assert binary.exists(), binary
        continue
    cmd = ["pixi", "run", "mojo", "build", "-I", ".", "-D", "MOJOLEARN_RF_BENCH_BINARY=1"]
    if a.mode == "identical": cmd += ["-D", "MOJOLEARN_NUMERIC_IDENTICAL=1"]
    if variant != "reference": cmd += ["-D", "MOJOLEARN_RF_HIST_" + variant.upper() + "=1"]
    cmd += ["ensemble/bench/rf_bench.mojo", "-o", str(binary)]
    with (out / (binary.name + ".build.log")).open("w") as log:
        subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    print("BUILT", binary.name, flush=True)
provenance = {
    "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    "binary_sha256": {v: hashlib.sha256((bin_dir / (a.mode + "_" + v)).read_bytes()).hexdigest() for v in variants},
    "command": vars(a) | {"output_dir": str(out), "binary_dir": str(bin_dir)},
}
(out / f"provenance_{a.mode}.json").write_text(json.dumps(provenance, indent=2) + "\n")
if a.build_only: raise SystemExit()
if a.prewarm:
    for variant in variants:
        binary = bin_dir / (a.mode + "_" + variant)
        r = subprocess.run([str(binary), "--tune", "clf", str(a.rows), str(a.cols)], cwd=ROOT, text=True, capture_output=True, check=True)
        (out / f"{binary.name}_{a.rows}_prewarm.log").write_text(r.stdout + r.stderr)
        print("PREWARM", variant, flush=True)
samples, hashes, canaries = {}, {}, []
per_pass = []
for index, variant in enumerate(variants + variants[::-1]):
    binary = bin_dir / (a.mode + "_" + variant)
    r = subprocess.run([str(binary), "--tune", "clf", str(a.rows), str(a.cols)], cwd=ROOT, text=True, capture_output=True, check=True)
    (out / f"{binary.name}_{a.rows}_{index}.log").write_text(r.stdout + r.stderr)
    expected_mode = "1" if a.mode == "identical" else "0"
    expected_tile = "0" if variant == "reference" else variant[-1]
    assert r.stdout.splitlines()[0].split()[1] == expected_mode, "wrong numeric mode"
    assert f"HIST_COLUMNS {expected_tile} classes 2" in r.stdout.splitlines(), "wrong tile or class configuration"
    vals = [float(line.split()[-1]) for line in r.stdout.splitlines() if line.startswith("ARM ")]
    assert len(vals) == 5 and all(v > 0 for v in vals), "missing or invalid fit samples"
    assert sum(line.startswith("MODEL ") for line in r.stdout.splitlines()) == 5, "missing model fingerprints"
    samples.setdefault(variant, []).extend(vals[1:])
    hashes.setdefault(variant, set()).update(line for line in r.stdout.splitlines() if line.startswith("MODEL "))
    canaries.extend(float(line.split()[2]) for line in r.stdout.splitlines() if line.startswith("CANARY ") and "warmup" not in line)
    per_pass.append({"index": index, "variant": variant, "median_ms": statistics.median(vals[1:]), "samples_ms": vals[1:]})
    print("DONE", index, variant, vals, flush=True)
assert all(len(v) == 1 for v in hashes.values()), "missing or unstable full-model fingerprints"
assert len(set.union(*hashes.values())) == 1, "candidate full-model fingerprint differs"
summary = {"mode": a.mode, "rows": a.rows, "cols": a.cols, "samples_ms": samples,
           "medians_ms": {k: statistics.median(v) for k,v in samples.items()},
           "passes": per_pass, "classes": 2, "hash_lines": {k: sorted(v) for k,v in hashes.items()}, "canaries": canaries,
           "canary_spread": max(canaries)/min(canaries), "timing_valid": max(canaries)/min(canaries) <= 1.1}
(out / f"timing_{a.mode}_{a.rows}.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
