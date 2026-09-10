#!/usr/bin/env python3
"""Summarize retained large kNN probes without automatically approving a default.

Usage: python tools/knn_probe_summary.py /path/to/probe > summary.json
Both execution orders, full outputs, and individual timed samples are required.
Phase timing is explicitly separated from ordinary end-to-end measurement.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


SHAPES = ((400000, 4000, 32, 10), (400000, 4000, 32, 15),
          (400000, 1000, 8, 15), (65537, 129, 17, 10))


def summarize(root):
    rows = []
    phase_mode = None
    shape_hashes = {}
    for n, q, d, k in SHAPES:
        for order in range(2):
            tag = f"n{n}-q{q}-d{d}-k{k}-p{order}"
            arms = {}
            for arm in ("default", "control"):
                path = root / f"{tag}-{arm}.log"
                lines = path.read_text().splitlines()
                headers = [s.split()[1:] for s in lines if s.startswith("KNN_REF_HEADER ")]
                results = [s.split()[1:] for s in lines if s.startswith("KNN_REF_RESULT ")]
                if len(headers) != 1 or len(results) != 1 or "KNN REFERENCE PRICE PASS" not in lines:
                    raise ValueError(f"incomplete run: {path}")
                header = dict(zip(headers[0][::2], headers[0][1::2]))
                result = dict(zip(results[0][::2], results[0][1::2]))
                for key, expected in (("index", n), ("queries", q), ("features", d), ("k", k)):
                    if int(header[key]) != expected or int(result[key]) != expected:
                        raise ValueError(f"wrong shape: {path}")
                rounds = int(result["rounds"])
                samples = {region: [] for region in ("request", "device")}
                phases = []
                for line in lines:
                    words = line.split()
                    if words[:1] == ["KNN_REF_ROUND"]:
                        samples[words[1]].append(float(words[3]))
                    if words[:2] == ["KNN_PHASE_TIMERS", "distance_ms"]:
                        phases.append({words[i]: float(words[i + 1]) for i in range(1, len(words), 2)})
                instrumented = any(s.startswith("KNN_PHASE_TIMERS ") for s in lines)
                if phase_mode is not None and phase_mode != instrumented:
                    raise ValueError("mixed instrumented and ordinary runs")
                phase_mode = instrumented
                medians = {}
                drift = {}
                for region, values in samples.items():
                    if len(values) != rounds or rounds < 3 or any(not (0 < x < float("inf")) for x in values):
                        raise ValueError(f"missing/invalid timed samples: {path}")
                    medians[region] = statistics.median(values)
                    if abs(medians[region] - float(result[f"{region}_median_ms"])) > 0.002:
                        raise ValueError(f"median disagrees with raw samples: {path}")
                    drift[region] = values[-1] / values[0]
                blob = (root / f"{tag}-{arm}.bin").read_bytes()
                if len(blob) != q * k * 8:
                    raise ValueError(f"incomplete output dump: {path}")
                arms[arm] = dict(samples_ms=samples, median_ms=medians,
                                 last_over_first=drift,
                                 full_output_sha256=hashlib.sha256(blob).hexdigest())
                if phases:
                    arms[arm]["phase_median_ms_including_warmups"] = {
                        key: statistics.median(row[key] for row in phases)
                        for key in ("distance_ms", "select_ms", "merge_ms")}
            if arms["default"]["full_output_sha256"] != arms["control"]["full_output_sha256"]:
                raise ValueError(f"full output mismatch: {tag}")
            digest = arms["default"]["full_output_sha256"]
            shape_key = (n, q, d, k)
            if shape_key in shape_hashes and shape_hashes[shape_key] != digest:
                raise ValueError(f"output moved between execution orders: {tag}")
            shape_hashes[shape_key] = digest
            rows.append(dict(shape=dict(index=n, queries=q, features=d, k=k), order=order,
                             first_arm="default" if order == 0 else "control", arms=arms,
                             default_over_control={region: arms["default"]["median_ms"][region] /
                                                   arms["control"]["median_ms"][region]
                                                   for region in ("request", "device")}))
    return dict(scope="phase diagnostic" if phase_mode else "ordinary request/device timing",
                all_large_targets_present=True, paired_full_outputs_match=True,
                default_decision="not automated; review target results, drift, and numerical evidence",
                rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.directory), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
