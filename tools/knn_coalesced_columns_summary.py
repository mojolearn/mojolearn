#!/usr/bin/env python3
"""Validate the complete coalesced-column A/B artifact and emit raw-backed JSON.

Usage: python tools/knn_coalesced_columns_summary.py /path/to/probe > summary.json
No GPU execution. Requires both passes/all five shapes, exact full dumps, all
five request/device samples, and passing component/public gates. The reported
ratios are ours/candidate A/B only; this tool does not invent opponent prices.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

SHAPES = ((400000, 4000, 32, 10), (400000, 4000, 32, 15),
          (400000, 1000, 8, 10), (10000, 32, 32, 15), (65537, 129, 17, 10))
ARMS = ("baseline", "coalesced")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_arm(root, tag, arm, shape):
    path = root / f"{tag}-{arm}.log"
    lines = path.read_text().splitlines()
    require("KNN REFERENCE PRICE PASS" in lines, f"missing PASS: {path}")
    require("KNN_REF_DEVICE_VS_REQUEST mismatched_cells 0" in lines,
            f"device/request comparison failed or missing: {path}")
    rows = [line.split()[1:] for line in lines if line.startswith("KNN_REF_RESULT ")]
    require(len(rows) == 1, f"expected one result: {path}")
    result = dict(zip(rows[0][::2], rows[0][1::2]))
    n, q, d, k = shape
    require(tuple(int(result[key]) for key in ("index", "queries", "features", "k")) == shape,
            f"shape metadata mismatch: {path}")
    require(int(result["rounds"]) == 5, f"wrong round count: {path}")
    timings = {}
    for phase in ("request", "device"):
        samples = [line.split() for line in lines if line.startswith(f"KNN_REF_ROUND {phase} ")]
        require([int(row[2]) for row in samples] == list(range(1, 6)), f"missing/duplicate samples: {path}/{phase}")
        values = [float(row[3]) for row in samples]
        require(all(0 < value < float("inf") for value in values), f"invalid timing: {path}/{phase}")
        timings[phase] = values
    blob_path = root / f"{tag}-{arm}.bin"
    blob = blob_path.read_bytes()
    require(len(blob) == q * k * 8, f"incomplete full-word dump: {blob_path}")
    return {"log": path.name, "samples_ms": timings,
            "full_output_sha256": hashlib.sha256(blob).hexdigest(),
            "reported_result": result}, blob


def summarize(root):
    for arm in ARMS:
        gate = (root / f"{arm}-gate.log").read_text()
        require("COALESCED DISTANCE PASS" in gate, f"component gate missing: {arm}")
        public_log = (root / f"{arm}-public-gate.log").read_text()
        require("KNN LAYOUT PUBLIC DISPATCH PASS" in public_log, f"public gate incomplete: {arm}")
        cells = (root / f"{arm}-public.cells").read_bytes()
        require(bool(cells), f"public gate cells empty: {arm}")
    require((root / "baseline-public.cells").read_bytes() == (root / "coalesced-public.cells").read_bytes(),
            "public cell outputs differ")
    report = {"comparison": "same-device baseline versus opt-in coalesced columns",
              "compiler": (root / "compiler.txt").read_text().strip(),
              "source": (root / "source.txt").read_text().strip(), "shapes": []}
    for shape in SHAPES:
        n, q, d, k = shape
        entry = {"index": n, "queries": q, "features": d, "k": k, "passes": []}
        combined = {arm: {phase: [] for phase in ("request", "device")} for arm in ARMS}
        previous = None
        for trial in range(2):
            tag = f"n{n}-q{q}-d{d}-k{k}-p{trial}"
            arms, blobs = {}, {}
            for arm in ARMS:
                arms[arm], blobs[arm] = read_arm(root, tag, arm, shape)
                for phase, samples in arms[arm]["samples_ms"].items():
                    combined[arm][phase].extend(samples)
            require(blobs["baseline"] == blobs["coalesced"], f"A/B full-word mismatch: {tag}")
            require(previous is None or previous == blobs["baseline"], f"output moved between passes: {tag}")
            previous = blobs["baseline"]
            entry["passes"].append({"order": list(ARMS if trial == 0 else reversed(ARMS)), "arms": arms})
        entry["pooled_median_ms"] = {arm: {phase: statistics.median(values) for phase, values in phases.items()}
                                      for arm, phases in combined.items()}
        entry["candidate_over_baseline"] = {
            phase: entry["pooled_median_ms"]["coalesced"][phase] / entry["pooled_median_ms"]["baseline"][phase]
            for phase in ("request", "device")}
        report["shapes"].append(entry)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.directory), indent=2))
