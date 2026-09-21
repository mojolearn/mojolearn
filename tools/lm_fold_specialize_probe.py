#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Record and gate the exact LM GEMM fold-stack specialization trial.

The timed worker is ``lm_step_memory_probe.py``.  This wrapper records the
fresh-process arm/outer provenance and summarizes exactly Taxi and Istella,
three alternating A/B process pairs apiece.  The first complete training step
is warmup; five subsequent public ``train_step`` calls are retained.
"""
import argparse
import glob
import hashlib
import json
import os
import statistics


DATASETS = ("taxi", "istella")
EXPECTED_SOURCE_SHA256 = {
    "taxi": "10d5d35f376a5b2ad5c66fabe624ee2caafb58bc5e7516824f801de9aab6cc15",
    "istella": "31f042376c0b819fe169cbbd998e840567c23dae25c14cb9054b460292f6ffef",
}
TARGET_SHAPE = [1, 2048, 768, 12, 12, 64, 2048, 12, 50257]
TARGET_PARAMETERS = 162147840
REQUIRED_OUTERS = 3
RETAINED = 5
DIAGNOSTIC_SPREAD_LIMIT = 1.10


def _sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _same(values):
    return len({json.dumps(v, sort_keys=True, allow_nan=False) for v in values}) == 1


def cmd_record(args):
    result = json.load(open(args.result))
    setup = None
    ids_sha256 = []
    with open(args.events) as stream:
        for line in stream:
            event = json.loads(line)
            if event.get("event") == "setup":
                setup = event
            elif event.get("event") == "step_start":
                ids_sha256.append(event.get("ids_sha256"))
    if setup is None:
        raise SystemExit("missing setup event")
    binding = os.path.abspath(args.binding)
    hardware = os.path.abspath(args.hardware)
    record = {
        "dataset": args.dataset,
        "arm": args.arm,
        "outer": args.outer,
        "launch_position": args.launch_position,
        "commit": args.commit,
        "binding": {"path": binding, "bytes": os.path.getsize(binding),
                    "sha256": _sha_file(binding)},
        "hardware": {"path": hardware, "bytes": os.path.getsize(hardware),
                     "sha256": _sha_file(hardware)},
        "runtime": setup.get("runtime"),
        "numeric_mode_env": setup.get("numeric_mode_env"),
        "input": {"seed": setup.get("seed"),
                  "initial_parameters_sha256": setup.get("initial_parameters_sha256"),
                  "tokens_per_step": setup.get("tokens_per_step"),
                  "ids_sha256": ids_sha256},
        "result": result,
    }
    with open(args.json, "w") as stream:
        json.dump(record, stream, indent=1, sort_keys=True, allow_nan=False)


def _trajectory(record):
    result = record["result"]
    return {"losses": result.get("step_losses"),
            "witnesses": result.get("step_witnesses")}


def _complete_witnesses(result):
    witnesses = result.get("step_witnesses", [])
    losses = result.get("step_losses", [])
    required = {"loss", "gradients", "parameters", "m", "v", "flags"}
    return (len(witnesses) == RETAINED + 1 and len(losses) == RETAINED + 1
            and [w.get("step") for w in witnesses] == list(range(1, RETAINED + 2))
            and [w.get("completed_steps") for w in witnesses] == list(range(1, RETAINED + 2))
            and all(set(w.get("sha256", {})) == required for w in witnesses))


def _complete_witnesses_one(result):
    witnesses = result.get("step_witnesses", [])
    required = {"loss", "gradients", "parameters", "m", "v", "flags"}
    return (len(witnesses) == 1 and len(result.get("step_losses", [])) == 1
            and witnesses[0].get("step") == 1
            and witnesses[0].get("completed_steps") == 1
            and set(witnesses[0].get("sha256", {})) == required)


def cmd_summarize(args):
    records = [json.load(open(path)) for pattern in args.files
               for path in sorted(glob.glob(pattern))]
    cells = {}
    failures = []
    for rec in records:
        dataset = rec.get("dataset")
        arm = rec.get("arm")
        if dataset not in DATASETS or arm not in ("baseline", "candidate", "sabotage"):
            failures.append("unexpected record cell %r/%r" % (dataset, arm))
            continue
        cells.setdefault(dataset, {}).setdefault(arm, []).append(rec)
    if set(cells) != set(DATASETS):
        failures.append("datasets=%r expected=%r" % (sorted(cells), list(DATASETS)))

    rows = []
    for dataset in DATASETS:
        arms = cells.get(dataset, {})
        row = {"dataset": dataset, "arms": {}}
        for arm in ("baseline", "candidate"):
            recs = sorted(arms.get(arm, []), key=lambda r: r.get("outer", -1))
            positions = {1: 0, 2: 1, 3: 0} if arm == "baseline" else {1: 1, 2: 0, 3: 1}
            structure_ok = (
                [r.get("outer") for r in recs] == [1, 2, 3]
                and all(r.get("launch_position") == positions[r["outer"]] for r in recs)
                and all(r["result"].get("shape") == TARGET_SHAPE for r in recs)
                and all(r["result"].get("parameters") == TARGET_PARAMETERS for r in recs)
                and all(r["result"].get("steps_completed") == RETAINED + 1 for r in recs)
                and all(len(r["result"].get("steady_step_seconds", [])) == RETAINED for r in recs)
                and all(_complete_witnesses(r["result"]) for r in recs)
                and all(len(r.get("input", {}).get("ids_sha256", [])) == RETAINED + 1 for r in recs)
                and all(r["result"].get("limited") is False for r in recs)
                and all(r.get("numeric_mode_env") == "identical" for r in recs)
            )
            medians = [statistics.median(r["result"]["steady_step_seconds"]) for r in recs]
            times = [v for r in recs for v in r["result"].get("steady_step_seconds", [])]
            record_spreads = [max(r["result"]["steady_step_seconds"]) /
                              min(r["result"]["steady_step_seconds"]) for r in recs]
            pooled_spread = max(times) / min(times) if times and min(times) > 0 else float("inf")
            internally_exact = _same([_trajectory(r) for r in recs])
            selection_ok = all(r["result"].get("gemm_fold_specialized") is (arm == "candidate")
                               for r in recs)
            valid = len(recs) == REQUIRED_OUTERS and structure_ok and internally_exact and selection_ok
            row["arms"][arm] = {
                "records": len(recs), "retained": len(times),
                "outer_medians_seconds": medians,
                "pooled_spread": pooled_spread, "record_spreads": record_spreads,
                "stable_diagnostic": (pooled_spread <= DIAGNOSTIC_SPREAD_LIMIT and
                                      all(v <= DIAGNOSTIC_SPREAD_LIMIT for v in record_spreads)),
                "internally_exact": internally_exact, "selection_ok": selection_ok,
                "valid": valid,
            }
            if not valid:
                failures.append("%s %s record/shape/exactness/selection invalid" % (dataset, arm))

        both = arms.get("baseline", []) + arms.get("candidate", [])
        provenance_ok = bool(both) and (
            _same([r.get("commit") for r in both])
            and _same([r.get("input") for r in both])
            and _same([r["result"].get("corpus") for r in both])
            and all(r["result"]["corpus"].get("sha256") == EXPECTED_SOURCE_SHA256[dataset]
                    for r in both)
            and all(str(r["result"]["corpus"].get("source_url", "")).startswith("r2://")
                    for r in both)
            and _same([r["binding"]["sha256"] for r in arms.get("baseline", [])])
            and _same([r["binding"]["sha256"] for r in arms.get("candidate", [])])
            and arms.get("baseline") and arms.get("candidate")
            and arms["baseline"][0]["binding"]["sha256"] != arms["candidate"][0]["binding"]["sha256"]
            and _same([(r.get("runtime") or {}).get("native_vendor") for r in both])
            and all((r.get("runtime") or {}).get("native_vendor") in ("cuda", "hip")
                    for r in both)
            and all((r.get("runtime") or {}).get("native_numeric_mode") == 1 for r in both)
            and _same([(r.get("runtime") or {}).get("source_sha256") for r in both])
            and _same([r.get("hardware", {}).get("sha256") for r in both])
            and all(r.get("hardware", {}).get("bytes", 0) > 0 for r in both)
        )
        ab_exact = bool(both) and _same([_trajectory(r) for r in both])
        bm = row["arms"]["baseline"]["outer_medians_seconds"]
        cm = row["arms"]["candidate"]["outer_medians_seconds"]
        ratio = (statistics.median(cm) / statistics.median(bm)
                 if bm and cm and statistics.median(bm) > 0 else float("inf"))
        conservative = max(cm) / min(bm) if bm and cm and min(bm) > 0 else float("inf")
        row.update(provenance_ok=provenance_ok, ab_exact=ab_exact,
                   median_of_process_medians_ratio=ratio,
                   conservative_ratio_diagnostic=conservative)
        sabotage = arms.get("sabotage", [])
        base0 = arms.get("baseline", [None])[0]
        sab0 = sabotage[0] if len(sabotage) == 1 else None
        sabotage_reached = bool(sab0 and base0) and (
            sab0.get("outer") == 1 and sab0.get("launch_position") == 0
            and sab0["result"].get("steps_completed") == 1
            and sab0["result"].get("steady_step_seconds") == []
            and _complete_witnesses_one(sab0["result"])
            and sab0["result"].get("gemm_fold_specialized") is True
            and sab0.get("commit") == base0.get("commit")
            and sab0.get("numeric_mode_env") == "identical"
            and sab0["result"].get("corpus") == base0["result"].get("corpus")
            and sab0.get("input", {}).get("initial_parameters_sha256") ==
                base0.get("input", {}).get("initial_parameters_sha256")
            and sab0.get("input", {}).get("ids_sha256") ==
                base0.get("input", {}).get("ids_sha256", [])[:1]
            and sab0.get("hardware", {}).get("sha256") ==
                base0.get("hardware", {}).get("sha256")
            and sab0["binding"]["sha256"] not in
                (base0["binding"]["sha256"], arms["candidate"][0]["binding"]["sha256"])
            and all(sab0["result"]["step_witnesses"][0]["sha256"][name] !=
                    base0["result"]["step_witnesses"][0]["sha256"][name]
                    for name in ("loss", "gradients", "parameters", "m", "v"))
            and sab0["result"]["step_losses"][0] != base0["result"]["step_losses"][0]
        )
        row["specialization_sabotage_reached"] = sabotage_reached
        row["pass"] = (row["arms"]["baseline"]["valid"] and
                       row["arms"]["candidate"]["valid"] and
                       provenance_ok and ab_exact and sabotage_reached and ratio < 1.0)
        if not provenance_ok:
            failures.append("%s provenance invalid" % dataset)
        if not ab_exact:
            failures.append("%s loss/state trajectory differs" % dataset)
        if not sabotage_reached:
            failures.append("%s specialization-only sabotage did not reach target step" % dataset)
        if ratio >= 1.0:
            failures.append("%s median-of-process-medians ratio %.6f is not faster" % (dataset, ratio))
        rows.append(row)
        print("LMFS GATE dataset=%s ratio=%.6f exact=%s provenance=%s pass=%s" %
              (dataset, ratio, ab_exact, provenance_ok, row["pass"]))

    summary = {
        "pass": not failures and all(row["pass"] for row in rows),
        "rule": ("exact losses/gradients/parameters/m/v/flags for every warmup and retained step; "
                 "candidate median-of-process-medians < baseline on both corpora; stability and "
                 "conservative extremes are recorded diagnostics"),
        "workload": {"shape": TARGET_SHAPE, "parameters": TARGET_PARAMETERS,
                     "tokens_per_step": TARGET_SHAPE[0] * TARGET_SHAPE[1],
                     "retained_steps_per_process": RETAINED},
        "cells": rows, "failures": failures,
    }
    with open(args.json, "w") as stream:
        json.dump(summary, stream, indent=1, sort_keys=True, allow_nan=False)
    if failures:
        print("LMFS REJECT " + "; ".join(failures))
        return 1
    print("LMFS PROMOTE Taxi and Istella exact full-step gates passed")
    return 0


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    rec = sub.add_parser("record")
    rec.add_argument("--dataset", choices=DATASETS, required=True)
    rec.add_argument("--arm", choices=("baseline", "candidate", "sabotage"), required=True)
    rec.add_argument("--outer", type=int, required=True)
    rec.add_argument("--launch-position", type=int, choices=(0, 1), required=True)
    rec.add_argument("--commit", required=True)
    rec.add_argument("--binding", required=True)
    rec.add_argument("--hardware", required=True)
    rec.add_argument("--result", required=True)
    rec.add_argument("--events", required=True)
    rec.add_argument("--json", required=True)
    rec.set_defaults(func=cmd_record)
    summary = sub.add_parser("summarize")
    summary.add_argument("files", nargs="+")
    summary.add_argument("--json", required=True)
    summary.set_defaults(func=cmd_summarize)
    args = parser.parse_args()
    return args.func(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
