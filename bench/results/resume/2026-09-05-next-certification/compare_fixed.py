"""Compare the retained fixed-source legs; never execute GPU work."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tools"))
from umap_identity_compare import read_capture


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_bits(entry):
    def flatten(values):
        for value in values:
            if isinstance(value, list):
                yield from flatten(value)
            else:
                yield value
    values = list(flatten(entry["uint32"]))
    require(len(values) == math.prod(entry["shape"]), "shape/byte inventory differs")
    require(all(type(v) is int and 0 <= v <= 0xffffffff
                and v & 0x7f800000 != 0x7f800000 for v in values), "invalid float32 bits")
    raw = struct.pack("<" + "I" * len(values), *values)
    require(hashlib.sha256(raw).hexdigest() == entry["float32_le_sha256"],
            "retained UMAP bytes do not match digest")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("amd", type=Path, help="AMD timestamped e1 directory")
    parser.add_argument("nvidia", type=Path, help="NVIDIA collected remote directory")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    amd, nv = args.amd / "diag", args.nvidia
    record = {"status": "RUNNING", "amd": str(args.amd), "nvidia": str(nv),
              "scope": "Named source fixtures on AMD and NVIDIA; no fresh Apple or installed-wheel claim",
              "mamba": {}, "umap_native": {}, "umap_quality": {}, "umap_identical": []}
    for profile, relative in (("baseline-v1", "mamba-cert"),
                              ("long-sequence-v1", "followup/mamba-long-cert")):
        result = subprocess.run([sys.executable, str(ROOT / "tools/mamba_backward_identity.py"),
                                 "compare", str(amd / relative), str(nv / relative)],
                                text=True, capture_output=True)
        record["mamba"][profile] = {"exit_code": result.returncode,
                                     "stdout": result.stdout, "stderr": result.stderr}
        if profile == "baseline-v1":
            require(result.returncode == 0, f"Mamba {profile}: {result.stderr}")
    for name in ("umap.identity.log", "followup/umap-broader.log"):
        left, right = read_capture(amd / name), read_capture(nv / name)
        require(left == right, f"native UMAP bits differ: {name}")
        record["umap_native"][name] = {"status": "PASS", "uint32_cells": len(left)}
    identical = []
    for mode, suffix in (("identical", ""), ("fast", "-fast"), ("deterministic", "-deterministic")):
        records = []
        for vendor, base in (("amd", amd), ("nvidia", nv)):
            d = json.loads((base / "followup" / f"transform-quality{suffix}.json").read_text())
            require(d["status"] == "PASS" and d["profile"] == "expanded"
                    and d["mode"] == mode and len(d["results"]) == 6
                    and all(r["passed"] for r in d["results"]), f"{vendor}/{mode}: quality incomplete")
            for row in d["results"]:
                for key in ("training_input", "query_input", "training_embedding", "query_embedding"):
                    validate_bits(row[key])
            record["umap_quality"][f"{vendor}/{mode}"] = {
                "status": "PASS", "cases": [{"profile": r["profile"], "quality": r["quality"],
                                               "control_margins": r["control_margins"]}
                                              for r in d["results"]]}
            records.append(d)
        a, b = records
        for key in ("schema", "profile", "harness_sha256", "thresholds", "k"):
            require(a[key] == b[key], f"quality contract differs: {key}")
        require(a["source"]["files_sha256"] == b["source"]["files_sha256"], "UMAP source hashes differ")
        if mode == "identical":
            identical = records
    for a, b in zip(identical[0]["results"], identical[1]["results"]):
        for key in ("profile", "parameters", "transform_schedule", "fitted_config", "fitted_mode",
                    "training_input", "query_input", "training_embedding", "query_embedding"):
            require(a[key] == b[key], f"IDENTICAL UMAP {a['profile']} differs: {key}")
        record["umap_identical"].append({"profile": a["profile"], "status": "PASS",
                                          "training_embedding_sha256": a["training_embedding"]["float32_le_sha256"],
                                          "query_embedding_sha256": a["query_embedding"]["float32_le_sha256"]})
    record["status"] = ("PASS" if all(v["exit_code"] == 0 for v in record["mamba"].values())
                        else "FAIL")
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print("PASS: baseline Mamba bytes; both UMAP stage fixtures; 36 held-out mode/vendor cases; six IDENTICAL embeddings")
    print("Long-sequence certificate:", "PASS" if record["status"] == "PASS" else "FAIL (see retained gate logs)")
    return 0 if record["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
