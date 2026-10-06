#!/usr/bin/env python3
"""Build both frozen I18 arms and compare complete forest output witnesses.

This is one device's correctness check, never a four-column quality receipt or
performance result. Compilation and execution retain the shared runner's logs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
TAGS = ("I18_FULL", "I18_SUBSAMPLED_FALLBACK", "I18_REGRESSION_FALLBACK")


def witnesses(log):
    found = {}
    for line in log.splitlines():
        words = line.split()
        if words and words[0] in TAGS:
            if len(words) != 3 or words[1] not in ("1", "2"):
                raise RuntimeError("malformed forest witness: " + line)
            key = (words[0], words[1])
            if key in found:
                raise RuntimeError("duplicate forest witness: " + line)
            found[key] = words[2]
    if set(found) != {(tag, stream) for tag in TAGS for stream in ("1", "2")}:
        raise RuntimeError("missing complete forest witnesses")
    if "I18 PASS mechanism permutations cache_bound reset full_forest_digests; compare OFF arm" not in log:
        raise RuntimeError("mechanism, byte-bound, or reset check did not complete")
    return {tag + "/streams=" + stream: value for (tag, stream), value in found.items()}


def require_frozen(expected, root=ROOT):
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    if actual != expected or dirty:
        raise RuntimeError("paired identity requires the requested clean frozen source")


def checked_stage_receipt(evidence, vendor, stage, arm, source, manifest_digest):
    path = evidence / ("I18-" + vendor + "-" + stage + "-" + arm + "-receipt.json")
    receipt = json.loads(path.read_text())
    expected = dict(schema=1, id="I18", mode="identical", vendor=vendor, stage=stage,
                    arm=arm, source_sha=source, manifest_sha256=manifest_digest,
                    status="COMPLETED", returncode=0)
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise RuntimeError("paired identity stage receipt mismatches " + key + ": " + str(path))
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                source_sha=receipt["source_sha"], manifest_sha256=receipt["manifest_sha256"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor", choices=("apple", "amd", "nvidia"), required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_sha
    require_frozen(source)
    manifest = ROOT / "experiments/performance_ideas/I18/manifest.json"
    manifest_digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    output = args.output.resolve()
    if output.is_relative_to(ROOT):
        parser.error("retain artifacts and evidence outside the source tree")
    output.mkdir(parents=True, exist_ok=False)
    packets = {}
    stage_receipts = {}
    for arm in ("baseline", "candidate"):
        binary = output / arm
        for stage in ("build", "validate"):
            evidence = output / (arm + "-" + stage)
            require_frozen(source)
            if hashlib.sha256(manifest.read_bytes()).hexdigest() != manifest_digest:
                raise RuntimeError("paired identity manifest changed before stage")
            command = [sys.executable, str(ROOT / "tools/performance_ideas.py"),
                       "execute", "I18", "--vendor", args.vendor, "--stage", stage,
                       "--arm", arm, "--output", str(binary), "--evidence", str(evidence)]
            with (output / (arm + "-" + stage + ".runner.log")).open("x") as stream:
                subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
            require_frozen(source)
            if hashlib.sha256(manifest.read_bytes()).hexdigest() != manifest_digest:
                raise RuntimeError("paired identity manifest changed during stage")
            stage_receipts[arm + "/" + stage] = checked_stage_receipt(
                evidence, args.vendor, stage, arm, source, manifest_digest)
            if stage == "validate":
                packets[arm] = witnesses((evidence / ("I18-" + args.vendor + "-validate-" + arm + ".log")).read_text())
    passed = packets["baseline"] == packets["candidate"]
    require_frozen(source)
    receipt = dict(id="I18", mode="identical", vendor=args.vendor, source_sha=source,
                   manifest_sha256=manifest_digest, stage_receipts=stage_receipts,
                   status="PASS_DEVICE_WITNESS" if passed else "FAIL_DEVICE_WITNESS",
                   witnesses=packets, four_column_identity="NOT_ESTABLISHED",
                   performance="NOT_MEASURED", promotion_authorized=False)
    (output / "device-witness.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("I18 paired device witness", receipt["status"], str(output / "device-witness.json"))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
