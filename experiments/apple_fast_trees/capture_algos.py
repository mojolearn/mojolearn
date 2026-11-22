#!/usr/bin/env python3
"""Capture the existing expanded-tree race as an Apple FAST full-workload cell.

Input preparation is a separate explicit future command:
  tools/bench_board_algos.py prep --full-tree-workload --lanes <tree lanes>
      --datasets taxi,istella --data <new directory>

This adapter always requests full-tree-workload and the existing whole-operation
timer. Historical capped cls/reg blocks cannot satisfy it. It never compiles,
downloads inputs, substitutes fixtures, measures opponents or promotes defaults.
Source only: uncompiled, unverified and unmeasured when written.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
LANES = (
    "decision-tree-clf", "decision-tree-reg", "bagging-clf", "bagging-reg",
    "adaboost-clf", "adaboost-reg", "dart", "dart-reg",
    "random-trees-embedding", "tree-shap",
)


def write_packet(path: Path, packet: dict) -> None:
    encoded = json.dumps(packet, indent=2, allow_nan=False) + "\n"
    with path.open("x") as stream:
        stream.write(encoded)


def captured_result(args, report: dict, artifacts: dict) -> dict:
    """Normalize retained records; never recompute model outputs or quality."""
    arm = report.get("arms", {}).get("ours-fast", {})
    if arm.get("status") != "ok" or len(arm.get("ms", [])) != 1:
        raise ValueError("expanded worker did not complete exactly one requested scored operation")
    captures = arm.get("aft_captures", [])
    receipts = arm.get("state_receipts", [])
    if len(captures) != 1 or len(receipts) != 1:
        raise ValueError("actual scored worker metadata or output receipt is missing")
    metadata, receipt = captures[0], receipts[0]
    if metadata.get("schema") != "aft-algos-capture-v1":
        raise ValueError("worker capture schema is missing")
    if metadata.get("mode") != "fast" or metadata.get("vendor") != "apple":
        raise ValueError("actual worker mode/vendor does not match Apple FAST")
    if not metadata.get("whole_operation_requested"):
        raise ValueError("constructor/model preparation was not inside the operation boundary")
    caps = metadata.get("intrinsic_caps", {})
    if (not caps.get("full_tree_workload") or caps.get("prepared_smoke_max_rows")
            or caps.get("applied_lane_subsets")):
        raise ValueError("full workload still has an intrinsic or requested row cap")
    prepared = metadata.get("prepared_block", {})
    dimensions = metadata.get("dimensions", {})
    if (prepared.get("full_tree_workload") is not True
            or dimensions != prepared.get("source_dimensions")
            or prepared.get("dataset") != args.dataset
            or metadata.get("dataset") != args.dataset):
        raise ValueError("actual worker rows/dataset do not match complete prepared source splits")
    if not metadata.get("input_array_sha256") or not prepared.get("source_array_sha256"):
        raise ValueError("full input provenance is missing")
    if args.lane == "tree-shap":
        tree_shap = metadata.get("estimator_settings", {}).get("tree_shap", {})
        if (tree_shap.get("background_rows") != dimensions["train"][0]
                or tree_shap.get("model_fit_rows") != dimensions["train"][0]
                or tree_shap.get("explain_rows") != dimensions["test"][0]):
            raise ValueError("TreeSHAP model/background/query rows are not the full declared input")
    scored = [operation for operation in arm.get("operations", []) if not operation.get("warmup")]
    if len(scored) != 1 or scored[0].get("scope") != "prepare-fit-consume":
        raise ValueError("complete preparation-fit-consume operation timing is absent")
    if metadata.get("pre_clock_fit") and scored[0].get("preparation_ms") is None:
        raise ValueError("pre-clock model fitting has no retained whole-operation preparation time")
    if receipt.get("output_status") != "ok" or not receipt.get("output_sha256"):
        raise ValueError("scored worker did not consume/hash its actual outputs")
    metrics = report.get("quality", {}).get("ours-fast", {})
    if not metrics or metrics.get("error"):
        raise ValueError("existing full held-out output metrics are missing or failed")
    output = report.get("saved_outputs", {}).get("ours-fast")
    if not output or not Path(output).is_file():
        raise ValueError("actual scored output artifact is missing")
    if not metadata.get("loaded_bindings"):
        raise ValueError("actual worker native-library provenance is missing")
    inference = report.get("infer", {}).get("arms", {}).get("ours-fast")
    if args.lane != "tree-shap" and (
        not inference or inference.get("status") != "ok" or len(inference.get("ms", [])) != 1
    ):
        raise ValueError("required inference/transform operation did not finish")
    packet = dict(
        metadata, contract="aft-full-workload-v1", status="CAPTURED", lane=args.lane,
        requested_dataset=args.dataset, output_sha256=receipt["output_sha256"],
        state_receipt=receipt, metrics=metrics, quality_status="pending",
        promotion_authorized=False, saved_outputs=output, **artifacts,
        timings={"whole_operation": scored, "fit_or_explanation_ms": arm["ms"],
                 "inference_or_transform_ms": arm.get("infer_ms", []),
                 "excluded_internal_warmup_ms": arm.get("warmup_ms"),
                 "excluded_internal_inference_warmup_ms": arm.get("infer_warmup_ms")},
        warmup_policy=report.get("aft_warmup_policy"),
        process_policy="outer queue phases use fresh isolated processes; startup construction/probe is excluded and retained in logs",
        phase_coverage={
            "whole_prepare_fit_consume": "captured",
            "preparation": "separate preparation_ms inside whole operation",
            "fit_or_explanation": "separate existing worker ms; includes explainer construction for TreeSHAP",
            "inference_or_transform": "separate existing worker infer_ms where supported",
            "fresh_process_cold_start": "not admitted: startup probe/build precedes operation",
            "repeated_model_inference_or_explanation": "pending separate matched full-workload recipe; this operation rebuilds the model",
        },
        dispatch_status="source mapping only; actual native hashes retained, kernel entry/quality acceptance pending",
    )
    if receipt.get("model", {}).get("status") == "ok":
        packet["model_sha256"] = receipt["model"]["sha256"]
    return packet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", required=True, choices=LANES)
    parser.add_argument("--dataset", required=True, choices=("taxi", "istella"))
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ready-seconds", type=float, default=3600)
    parser.add_argument("--round-seconds", type=float, default=3600)
    args = parser.parse_args()
    if (os.environ.get("MOJOLEARN_AFT_CAPTURE") != "1"
            or os.environ.get("MOJOLEARN_BENCH_INSTALLED") != "1"):
        raise ValueError("use the isolated Apple FAST tree pipeline worker")
    if (os.environ.get("MOJOLEARN_NUMERIC_MODE") != "fast"
            or os.environ.get("MOJOLEARN_VENDOR") != "apple"):
        raise ValueError("expanded tree capture requires explicit Apple FAST")
    for key in ("MOJOLEARN_ALGOS_SMOKE_ROWS", "MOJOLEARN_BENCH_DATA_ROWS"):
        if os.environ.get(key):
            raise ValueError("full tree capture refuses " + key)
    if args.output.exists():
        raise FileExistsError("retain prior capture evidence: " + str(args.output))
    evidence = args.output.with_suffix(".algos")
    evidence.mkdir(parents=True, exist_ok=False)
    race_dir, work_dir = evidence / "race", evidence / "outputs"
    log = evidence / "conductor.log"
    argv = [
        sys.executable, str(ROOT / "tools/bench_board_algos.py"), "race",
        "--lane", args.lane, "--dataset", args.dataset, "--data", str(args.data.resolve()),
        "--arms", "ours-fast", "--rounds", "1", "--full-tree-workload",
        "--out", str(race_dir.resolve()), "--work", str(work_dir.resolve()),
        "--ours-python", sys.executable, "--ready-seconds", str(args.ready_seconds),
        "--warmup-seconds", str(args.round_seconds), "--round-seconds", str(args.round_seconds),
    ]
    env = os.environ.copy()
    env["MOJOLEARN_BENCH_WHOLE_OPERATION"] = "1"
    env["MOJOLEARN_AFT_FULL_TREE_WORKLOAD"] = "1"
    artifacts = {"full_log": str(log.resolve()), "evidence_directory": str(evidence.resolve()),
                 "worker_logs": str(race_dir.resolve()), "worker_command": argv}
    with log.open("x") as stream:
        rc = subprocess.run(argv, env=env, stdout=stream, stderr=subprocess.STDOUT).returncode
    report_path = race_dir / (args.lane + "-" + args.dataset + ".json")
    report = None
    try:
        if report_path.is_file():
            report = json.loads(report_path.read_text())
        if rc or report is None:
            raise RuntimeError("expanded full workload incomplete, rc=%s; retain %s" % (rc, log))
        artifacts["race_result"] = str(report_path.resolve())
        packet = captured_result(args, report, artifacts)
        write_packet(args.output, packet)
    except Exception as exc:
        write_packet(args.output, {
            "contract": "aft-full-workload-v1", "status": "INCOMPLETE", "lane": args.lane,
            "requested_dataset": args.dataset, "returncode": rc, "error": str(exc),
            "quality_status": "pending", "promotion_authorized": False,
            "race_result": str(report_path.resolve()), **artifacts,
        })
        raise


if __name__ == "__main__":
    main()
