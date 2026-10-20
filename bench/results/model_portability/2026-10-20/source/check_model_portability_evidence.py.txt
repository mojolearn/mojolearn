#!/usr/bin/env python3
"""Check retained custom-model evidence without running or timing a model.

Usage: python3 tools/check_model_portability_evidence.py EVIDENCE_DIR [--write-report]
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

VENDORS = ("metal", "cuda", "hip")
MODELS = {
    "mnist-12", "mobilenet_v3_small", "resnet18",
    "resnet50-v1-12", "smollm2-135m", "squeezenet1_1",
}
SOURCE_PINS = (
    "weights", "weights_url", "repo", "revision", "url", "sha256",
    "converted_sha256",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def review(root):
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    require(manifest["schema"] == "mojolearn-model-portability/1", "manifest schema")
    for name, record in manifest["files"].items():
        data = (root / name).read_bytes()
        require(len(data) == record["bytes"], f"{name}: byte count")
        require(hashlib.sha256(data).hexdigest() == record["sha256"], f"{name}: SHA-256")

    def retained(name):
        require(name in manifest["files"], f"unbound evidence file: {name}")
        return root / name

    training = {}
    for vendor in VENDORS:
        text = retained(manifest["training"][vendor]["log"]).read_text()
        tail = text.splitlines()[-1]
        require(re.search(r"\b\d+ passed\b", tail), f"{vendor}: no completed test summary")
        require(not re.search(r"\b(?:failed|error|skipped|xfailed)\b", tail),
                f"{vendor}: incomplete or failed selected test run")
        found = {}
        for name, digest in re.findall(r"PUBLIC_DIGEST (\S+) ([0-9a-f]{16})", text):
            require(name not in found or found[name] == digest, f"{vendor}: repeated {name} differs")
            found[name] = digest
        training[vendor] = found
    training_matches = {}
    for name in manifest["selected_training_digests"]:
        values = [training[v].get(name) for v in VENDORS]
        require(None not in values and len(set(values)) == 1, f"training {name}: missing or unequal")
        training_matches[name] = values[0]

    models = {}
    for vendor in VENDORS:
        rows = {}
        for name in manifest["models"][vendor]:
            row = read_json(retained(name))
            require(row["schema"] == "mojolearn-real-model/1", f"{name}: schema")
            require(row["vendor"] == vendor and row["numeric_mode"] == "identical", f"{name}: mode/vendor")
            require(row["status"] == "ok" and row["ours"]["repeat_equal"] is True, f"{name}: failed/repeat")
            require(row["model"] not in rows, f"{vendor}: duplicate model")
            rows[row["model"]] = row
        require(set(rows) == MODELS, f"{vendor}: model coverage")
        models[vendor] = rows
    catalog = {m["name"]: m for m in read_json(retained(manifest["catalog"]))["models"]}
    model_matches = []
    for name in sorted(MODELS):
        rows = [models[v][name] for v in VENDORS]
        for field in ("batch", "seed", "input_digest", "input_shapes", "inputs", "outputs", "kind"):
            require(len({json.dumps(r[field], sort_keys=True) for r in rows}) == 1, f"{name}: {field}")
        for field in SOURCE_PINS:
            values = [r["source"][field] for r in rows if field in r["source"]]
            require(len({json.dumps(v, sort_keys=True) for v in values}) <= 1, f"{name}: source.{field}")
        digests = [r["ours"]["digest"] for r in rows]
        require(len(set(digests)) == 1, f"{name}: output mismatch")
        cell = catalog[name]["versions"][manifest["release_context"]]["identical"]
        require(all(cell["digests"].get(v) == digests[0] for v in VENDORS), f"{name}: catalog mismatch")
        require(catalog[name]["check"]["input_digest"] == rows[0]["input_digest"], f"{name}: catalog input")
        model_matches.append({
            "model": name, "input_shapes": rows[0]["input_shapes"],
            "input_digest": rows[0]["input_digest"], "output_digest": digests[0],
            "vendors": list(VENDORS),
        })

    saved = manifest["saved_program"]
    source = read_json(retained(saved["cuda"]))
    target = read_json(retained(saved["metal"]))
    run = read_json(retained(saved["run"]))
    require(run["exit_code"] == 0, "reload command failed")
    for vendor, row in (("cuda", source), ("metal", target)):
        require(row["format"] == "mojolearn-model-verify/1", f"{vendor}: receipt schema")
        require(row["install"]["vendor"] == vendor, f"{vendor}: receipt vendor")
        require(row["install"]["numeric_mode"] == "identical", f"{vendor}: receipt mode")
    for field in ("sha256", "description_hash", "bytes", "kind"):
        require(source["model"][field] == target["model"][field], f"saved model: {field}")
    for field in ("seed", "input_source", "inputs", "outputs"):
        require(source[field] == target[field], f"saved model: {field}")
    artifact = manifest["retained_model"]
    require(artifact["sha256"] == source["model"]["sha256"], "retained model hash")
    require(artifact["bytes"] == source["model"]["bytes"], "retained model size")
    retention = read_json(retained(artifact["retention_receipt"]))
    require(retention["download_sha256"] == artifact["sha256"], "R2 read-back hash")
    require(retention["r2_key"] == artifact["r2_key"], "R2 key")
    require(retention["upload_exit_code"] == retention["download_exit_code"] == 0, "R2 retention failed")
    return {
        "schema": "mojolearn-model-portability-review/1",
        "date": manifest["date"],
        "training_digests": training_matches,
        "imported_models": model_matches,
        "saved_program": {
            "model_sha256": artifact["sha256"],
            "input_digest": source["inputs"][0]["digest"],
            "output_digest": source["outputs"][0]["digest"],
            "source": source["install"], "destination": target["install"],
        },
        "performance_results_admitted": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    report = review(args.evidence)
    path = args.evidence / "review.json"
    if args.write_report:
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    else:
        require(read_json(path) == report, "review.json differs from retained evidence")
    print("PASS: %d training digests across three vendors; %d imported models across three vendors; "
          "NVIDIA-to-Apple saved-program match; retained artifact verified; no performance claim."
          % (len(report["training_digests"]), len(report["imported_models"])))


if __name__ == "__main__":
    main()
