#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build frozen GEMM A/B probes separately from GPU execution.

  python tools/identical_speed_ab.py plan
  python tools/identical_speed_ab.py build --vendor nvidia --arch sm_89 --out /path/build
  python tools/identical_speed_ab.py run --build /path/build --out /path/results
  python tools/identical_speed_ab.py run --build /path/apple-build --out /path/id --identity-only
  python tools/identical_speed_ab.py run --build /path/build --out /path/id --identity-only --identity-shapes light
  python tools/identical_speed_ab.py compare-identity --left /path/apple/id/results.json --right /path/nvidia/id/results.json --out /path/comparison.json

This is synthetic kernel screening, not a model speed claim or a default-flip
gate. One excluded warmup and one measured sample per arm, per owner policy.
Carry winners to both real datasets/corpora, both timing vendors and Apple/host
identity checks before promotion. Never compile on a timing GPU.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/identical_speed"
COLUMNS = {"nvidia": "nvidia", "amd": "amd", "apple": "apple"}
# Adjacent tile/leaf boundaries and all three orientation paths. The flat
# reference checks all cells, independently of the schedule under test.
CHECKS = [dict(name=f"bits-{m}-{n}-{k}-{op}-{tiny}", m=m, n=n, k=k,
               op=op, tiny=tiny, reference=1, timed=0)
          for m, n, k in [(31, 33, 127), (65, 67, 129), (129, 131, 257)]
          for op in range(3) for tiny in (0, 1)]
# Regular shapes and neighboring ragged shapes, never used as dispatch rules.
SCREENS = [dict(name=f"screen-{m}-{n}-{k}-{op}", m=m, n=n, k=k,
                op=op, tiny=0, reference=0, timed=1)
           for m, n, k in [(1024, 512, 768), (1025, 513, 769),
                           (2048, 768, 1024), (2049, 769, 1025),
                           (512, 256, 4096), (513, 257, 4097)]
           for op in range(3)]


def identity_cases(suite):
    """Light identity includes real dispatch paths, with every timer disabled.

    Tiny fixtures alone take the same fallback tile in several experiment
    arms. Add an aligned tile, a ragged tile and a long reduction, each in
    NN/NT/TN. These are test inputs, never dispatch thresholds. No model fit,
    dataset download, warmup/sample timing or performance verdict is involved.
    """
    if suite == "tiny":
        return CHECKS
    if suite != "light":
        raise ValueError("unknown identity suite")
    shapes = {(1024, 512, 768), (1025, 513, 769), (512, 256, 4096)}
    return CHECKS + [dict(c, name=c["name"].replace("screen-", "identity-"), timed=0)
                     for c in SCREENS if (c["m"], c["n"], c["k"]) in shapes]


def compare_identity(left, right):
    """Compare complete recorded output fingerprints; fail on missing coverage."""
    expected = None
    indexed = []
    for report in (left, right):
        if report.get("status") != "PASS" or report.get("scope") != "identity-only":
            raise ValueError("both reports must be complete, untimed identity runs")
        cases = identity_cases(report.get("identity_shapes", "tiny"))
        fixtures = {c["name"]: c for c in cases}
        profiles = report["build"]["profiles"]
        required = {(name, c["name"]) for name in profiles for c in cases}
        if report["expected_pairs"] != len(required) or len(report["pairs"]) != len(required):
            raise ValueError("incomplete identity coverage")
        rows = {}
        for pair in report["pairs"]:
            key = (pair["profile"], pair["case"]["name"])
            if fixtures.get(key[1]) != pair["case"]:
                raise ValueError("identity fixture does not match the declared suite")
            if key in rows or pair["bits"] != "MATCH" or pair["case"]["timed"] != 0:
                raise ValueError("duplicate, mismatched or timed identity case")
            if set(pair["arms"]) != {"baseline", pair["profile"]}:
                raise ValueError("missing baseline or candidate")
            if any(row["ns"] != 0 for row in pair["arms"].values()):
                raise ValueError("identity report contains timings")
            rows[key] = pair
        if set(rows) != required or (expected is not None and required != expected):
            raise ValueError("identity case sets differ")
        expected = required
        indexed.append(rows)
    if left["build"]["commit"] != right["build"]["commit"]:
        raise ValueError("compiled source commits differ")
    if left["build"]["vendor"] == right["build"]["vendor"]:
        raise ValueError("cross-vendor identity requires different vendors")
    outputs = 0
    for key in expected:
        a, b = (rows[key] for rows in indexed)
        if a["case"] != b["case"]:
            raise ValueError("fixture definitions differ")
        for arm in a["arms"]:
            x, y = a["arms"][arm], b["arms"][arm]
            if x["hash"] != y["hash"] or x["mask"] != y["mask"]:
                raise ValueError(f"cross-vendor output mismatch: {key} {arm}")
            outputs += 1
    return {"status": "PASS", "compiled_source": left["build"]["commit"],
            "vendors": [left["build"]["vendor"], right["build"]["vendor"]],
            "pairs": len(expected), "output_fingerprints_compared": outputs,
            "timed_samples": 0, "method": "FNV-1a64 over every output byte; tiny cases also compare every word to flat plan",
            "scope": "lightweight GEMM identity only; not full model or arbitrary toggle combinations"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def profiles():
    return json.loads((CONFIG / "profiles.json").read_text())["profiles"]


def selected(names):
    choices = profiles()
    selection = CONFIG / "selection.json"
    if not names and selection.exists():
        names = json.loads(selection.read_text())["profiles"]
    names = names or list(choices)
    if not names or len(set(names)) != len(names) or any(n not in choices for n in names):
        raise ValueError("unknown, empty or duplicate profile selection")
    return {n: choices[n] for n in names}


def parse_record(log, case, mask, vendor):
    lines = [s for s in log.splitlines() if s.startswith("SCHEDULE_RESULT ")]
    if len(lines) != 1:
        raise ValueError(f"expected one result, got {len(lines)}")
    record = dict(field.split("=", 1) for field in lines[0].split()[1:])
    if record.get("mode", "").lower() != "identical":
        raise ValueError("wrong compiled numeric mode")
    if record.get("column", "").lower() != COLUMNS[vendor]:
        raise ValueError("wrong compiled vendor column")
    for key, value in {**{k: case[k] for k in ("m", "n", "k", "op", "tiny", "reference", "timed")}, "mask": mask}.items():
        if int(record[key]) != value:
            raise ValueError(f"wrong {key}: {record[key]}, expected {value}")
    for key in ("ns", "hash", "workspace_bytes", "slack", "rpt", "cpt", "mask"):
        record[key] = int(record[key])
    if record["ns"] < 0 or (case["timed"] and record["ns"] == 0):
        raise ValueError("invalid synchronized duration")
    if not case["timed"] and record["ns"] != 0:
        raise ValueError("identity-only case unexpectedly timed")
    if not any(s.startswith("SCHEDULE_DEVICE ") for s in log.splitlines()):
        raise ValueError("missing device witness")
    record["device"] = next(s.removeprefix("SCHEDULE_DEVICE ") for s in log.splitlines() if s.startswith("SCHEDULE_DEVICE "))
    record["dispatch"] = next(s.removeprefix("SCHEDULE_DISPATCH ") for s in log.splitlines() if s.startswith("SCHEDULE_DISPATCH "))
    return record


def compare_pair(a, b, timed):
    if a["device"] != b["device"]:
        raise ValueError("A/B devices differ")
    if a["hash"] != b["hash"]:
        raise ValueError("A/B output hashes differ")
    return {"bits": "MATCH", "b_over_a": b["ns"] / a["ns"] if timed else None,
            "baseline_ns": a["ns"], "candidate_ns": b["ns"],
            "workspace_a": a["workspace_bytes"], "workspace_b": b["workspace_bytes"]}


def screen_summary(pairs):
    result = {}
    for name in sorted({p["profile"] for p in pairs}):
        rows = [p for p in pairs if p["profile"] == name and p["b_over_a"] is not None]
        if rows:
            ratios = [p["b_over_a"] for p in rows]
            result[name] = {"cases": len(rows), "applicable": all(p["applicable"] for p in rows),
                            "geomean_b_over_a": math.exp(sum(math.log(r) for r in ratios) / len(ratios)),
                            "min_b_over_a": min(ratios), "max_b_over_a": max(ratios),
                            "verdict": "SINGLE_SAMPLE_SCREEN_ONLY_NO_PROMOTION"}
    return result


def build(args):
    if git("status", "--porcelain", "--untracked-files=normal"):
        raise ValueError("commit the experiment first: builds require a clean frozen tree")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    candidates = selected(args.profiles)
    compiler = shlex.split(args.mojo)
    version = subprocess.check_output(compiler + ["--version"], cwd=ROOT, text=True).strip()
    arms = {"baseline": {"mask": 0, "define": None}, **candidates}
    manifest = {"schema": 1, "commit": git("rev-parse", "HEAD"),
                "vendor": args.vendor, "arch": args.arch, "compiler": version,
                "host": platform.platform(), "host_system": platform.system(),
                "host_machine": platform.machine(), "profiles": candidates, "arms": {}}
    for name, profile in arms.items():
        artifact = out / name
        cmd = compiler + ["build", "-j", "1", "-D", "MOJOLEARN_NUMERIC_IDENTICAL=1"]
        if args.vendor != "apple":
            if not args.arch or not args.arch.startswith("sm_" if args.vendor == "nvidia" else "gfx"):
                raise ValueError("specify the actual target architecture for this vendor")
            cmd += ["--target-accelerator", args.arch, "-D", "MOJOLEARN_COLUMN_" + args.vendor.upper()]
        if profile["define"]:
            cmd += ["-D", profile["define"] + "=1"]
        cmd += ["-I", str(ROOT), str(ROOT / "bench/gemm_schedule_ab.mojo"), "-o", str(artifact)]
        with (out / (name + ".build.log")).open("w") as log:
            result = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        row = {"command": cmd, "exit_code": result.returncode, "mask": profile["mask"]}
        if result.returncode == 0:
            row.update(file=name, sha256=sha(artifact))
        manifest["arms"][name] = row
        write_json(out / "build.json", manifest)
        print(f"BUILD profile={name} rc={result.returncode}", flush=True)
        if result.returncode:
            raise RuntimeError(f"compile failed; see {out / (name + '.build.log')}")
    manifest["status"] = "COMPILE_PASS"
    write_json(out / "build.json", manifest)


def run(args):
    manifest = json.loads((args.build / "build.json").read_text())
    if manifest.get("status") != "COMPILE_PASS":
        raise ValueError("only a complete green frozen build can run")
    if (manifest["host_system"], manifest["host_machine"]) != (platform.system(), platform.machine()):
        raise ValueError("host executable target differs: compile on a cheap host matching the GPU box")
    vendor = manifest["vendor"]
    if not args.identity_only and (vendor == "apple" or platform.system() == "Darwin"):
        raise ValueError("IDENTICAL timing runs on NVIDIA/AMD; use --identity-only on Apple")
    if set(manifest["arms"]) != {"baseline", *manifest["profiles"]}:
        raise ValueError("build manifest has missing or extra arms")
    for name, row in manifest["arms"].items():
        if row["exit_code"] != 0 or sha(args.build / row["file"]) != row["sha256"]:
            raise ValueError(f"unverified binary: {name}")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cases = identity_cases(args.identity_shapes) if args.identity_only else CHECKS + SCREENS
    report = {"schema": 1, "status": "RUNNING", "build": manifest,
              "scope": "identity-only" if args.identity_only else "synthetic-kernel-screen",
              "samples_per_arm": 0 if args.identity_only else 1,
              "identity_shapes": args.identity_shapes if args.identity_only else None,
              "runner_sha256": sha(__file__),
              "expected_pairs": len(cases) * len(manifest["profiles"]),
              "pairs": [], "failure": None}
    write_json(out / "results.json", report)
    try:
        for name, profile in manifest["profiles"].items():
            for index, case in enumerate(cases):
                records = {}
                # Alternate which arm runs first across cases to reduce order bias.
                for arm in (["baseline", name] if index % 2 == 0 else [name, "baseline"]):
                    row = manifest["arms"][arm]
                    env = {k: v for k, v in os.environ.items() if not k.startswith(("MOJOLEARN_GEMM_", "SCHEDULE_"))}
                    env.update({"SCHEDULE_" + key.upper(): str(value) for key, value in case.items() if key != "name"})
                    env["SCHEDULE_EXPECT_MASK"] = str(row["mask"])
                    log_path = out / f"{name}.{case['name']}.{arm}.log"
                    with log_path.open("w") as log:
                        child = subprocess.run([str((args.build / row["file"]).resolve())], env=env,
                                               stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
                    if child.returncode:
                        raise RuntimeError(f"probe failed rc={child.returncode}: {log_path}")
                    records[arm] = parse_record(log_path.read_text(), case, row["mask"], vendor)
                pair = compare_pair(records["baseline"], records[name], case["timed"])
                pair.update(profile=name, case=case, arms=records,
                            applicable=vendor in profile["vendors"],
                            verdict="SCREEN_ONLY" if case["timed"] else "BITS_PASS")
                report["pairs"].append(pair)
                write_json(out / "results.json", report)
                if case["timed"]:
                    print(f"SCREEN profile={name} case={case['name']} B/A={pair['b_over_a']:.4f} bits=MATCH", flush=True)
        report["status"] = "PASS"
    except Exception as exc:
        report["status"], report["failure"] = "FAILED", str(exc)
        raise
    finally:
        report["summary"] = screen_summary(report["pairs"])
        write_json(out / "results.json", report)
        print(f"AB status={report['status']} pairs={len(report['pairs'])}/{report['expected_pairs']} evidence={out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    subs.add_parser("plan")
    p = subs.add_parser("build")
    p.add_argument("--vendor", choices=COLUMNS, required=True)
    p.add_argument("--arch")
    p.add_argument("--profiles", nargs="+")
    p.add_argument("--mojo", default="pixi run --frozen mojo")
    p.add_argument("--out", type=Path, required=True)
    p = subs.add_parser("run")
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--identity-only", action="store_true")
    p.add_argument("--identity-shapes", choices=("tiny", "light"), default="tiny")
    p.add_argument("--timeout", type=int, default=300)
    p = subs.add_parser("compare-identity")
    p.add_argument("--left", type=Path, required=True)
    p.add_argument("--right", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        print(json.dumps({"base": git("rev-parse", "HEAD"), "profiles": selected(None),
                          "checks_per_arm": len(CHECKS), "screens_per_arm": len(SCREENS),
                          "status": "UNMEASURED"}, indent=2))
    elif args.command == "build":
        build(args)
    elif args.command == "compare-identity":
        report = compare_identity(json.loads(args.left.read_text()), json.loads(args.right.read_text()))
        report["reports"] = [{"path": str(p), "sha256": sha(p)} for p in (args.left, args.right)]
        write_json(args.out, report)
        print(json.dumps(report))
    else:
        run(args)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, subprocess.SubprocessError, OSError, KeyError) as exc:
        print(f"AB ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
