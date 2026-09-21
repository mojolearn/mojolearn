#!/usr/bin/env python3
"""Record and gate exact repeated training with attention block swizzling."""
import argparse
import glob
import hashlib
import json
import os
import statistics


DATASETS = ("taxi", "istella")
ARMS = ("baseline", "candidate")
BASE_ARM = "stash_tiled_fgrid_r32_qres_pf_estash_dres_kvgrid_r32"
SWZ_ARM = BASE_ARM + "_bswz"
EXPECTED_SOURCE_SHA256 = {
    "taxi": "10d5d35f376a5b2ad5c66fabe624ee2caafb58bc5e7516824f801de9aab6cc15",
    "istella": "31f042376c0b819fe169cbbd998e840567c23dae25c14cb9054b460292f6ffef",
}
TARGET_SHAPE = {
    "batch": 1, "length": 2048, "d_model": 768, "n_heads": 12,
    "n_kv": 12, "head_dim": 64, "intermediate": 2048,
    "n_layers": 12, "vocab_size": 50257,
}
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


def _trajectory(record):
    result = record["result"]
    return {"losses": result.get("step_losses"),
            "witnesses": result.get("step_witnesses")}


def _complete_witnesses(result, count):
    witnesses = result.get("step_witnesses", [])
    losses = result.get("step_losses", [])
    required = {"loss", "gradients", "parameters", "m", "v", "flags"}
    return (len(witnesses) == count and len(losses) == count
            and [w.get("step") for w in witnesses] == list(range(1, count + 1))
            and [w.get("completed_steps") for w in witnesses] == list(range(1, count + 1))
            and all(set(w.get("sha256", {})) == required for w in witnesses))


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
    record = {
        "dataset": args.dataset,
        "arm": args.arm,
        "outer": args.outer,
        "launch_position": args.launch_position,
        "commit": args.commit,
        "device_identity": {"path": os.path.abspath(args.device_identity),
                            "bytes": os.path.getsize(args.device_identity),
                            "sha256": _sha_file(args.device_identity)},
        "binding": {"path": binding, "bytes": os.path.getsize(binding),
                    "sha256": _sha_file(binding)},
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


def _route_ok(record, provider, arm):
    result = record["result"]
    expected = BASE_ARM if arm == "baseline" else SWZ_ARM
    default = SWZ_ARM if provider == "nvidia" else BASE_ARM
    return (result.get("attention_arm") == expected
            and result.get("attention_arm_resolved_hd64") == expected
            and result.get("attention_arm_default") == default
            and result.get("attention_arm_is_default") is (expected == default)
            and result.get("attention_arm_trial_build") is True
            and result.get("attention_arm_source") == "binding")


def cmd_summarize(args):
    records = [json.load(open(path)) for pattern in args.files
               for path in sorted(glob.glob(pattern))]
    provider = args.provider
    expected_vendor = "cuda" if provider == "nvidia" else "hip"
    timed = [r for r in records if r.get("arm") in ARMS]
    sabotage = [r for r in records if r.get("arm") == "sabotage"]
    cells = {}
    failures = []
    for rec in timed:
        dataset = rec.get("dataset")
        if dataset not in DATASETS:
            failures.append("unexpected dataset %r" % dataset)
            continue
        cells.setdefault(dataset, {}).setdefault(rec["arm"], []).append(rec)
    if set(cells) != set(DATASETS):
        failures.append("datasets=%r expected=%r" % (sorted(cells), list(DATASETS)))

    rows = []
    for dataset in DATASETS:
        arms = cells.get(dataset, {})
        row = {"dataset": dataset, "arms": {}}
        for arm in ARMS:
            recs = sorted(arms.get(arm, []), key=lambda r: r.get("outer", -1))
            positions = {1: 0, 2: 1, 3: 0} if arm == "baseline" else {1: 1, 2: 0, 3: 1}
            structure_ok = (
                [r.get("outer") for r in recs] == [1, 2, 3]
                and all(r.get("launch_position") == positions[r["outer"]] for r in recs)
                and all(r["result"].get("shape") == TARGET_SHAPE for r in recs)
                and all(r["result"].get("parameters") == TARGET_PARAMETERS for r in recs)
                and all(r["result"].get("steps_completed") == RETAINED + 1 for r in recs)
                and all(len(r["result"].get("steady_step_seconds", [])) == RETAINED for r in recs)
                and all(_complete_witnesses(r["result"], RETAINED + 1) for r in recs)
                and all(len(r.get("input", {}).get("ids_sha256", [])) == RETAINED + 1 for r in recs)
                and all(r["result"].get("limited") is False for r in recs)
                and all(r.get("numeric_mode_env") == "identical" for r in recs)
                and all(_route_ok(r, provider, arm) for r in recs))
            medians = [statistics.median(r["result"]["steady_step_seconds"]) for r in recs]
            times = [v for r in recs for v in r["result"].get("steady_step_seconds", [])]
            spreads = [max(r["result"]["steady_step_seconds"]) /
                       min(r["result"]["steady_step_seconds"]) for r in recs]
            pooled = max(times) / min(times) if times and min(times) > 0 else float("inf")
            internally_exact = _same([_trajectory(r) for r in recs])
            valid = len(recs) == REQUIRED_OUTERS and structure_ok and internally_exact
            row["arms"][arm] = {
                "records": len(recs), "retained": len(times),
                "outer_medians_seconds": medians,
                "record_spreads": spreads, "pooled_spread": pooled,
                "stable_diagnostic": (pooled <= DIAGNOSTIC_SPREAD_LIMIT and
                                      all(v <= DIAGNOSTIC_SPREAD_LIMIT for v in spreads)),
                "internally_exact": internally_exact, "route_ok": structure_ok,
                "valid": valid,
            }
            if not valid:
                failures.append("%s %s record/shape/exactness/route invalid" % (dataset, arm))

        both = arms.get("baseline", []) + arms.get("candidate", [])
        provenance_ok = bool(both) and (
            _same([r.get("commit") for r in both])
            and _same([r.get("device_identity", {}).get("sha256") for r in both])
            and all(r.get("device_identity", {}).get("bytes", 0) > 0 for r in both)
            and _same([r.get("input") for r in both])
            and _same([r["result"].get("corpus") for r in both])
            and all(r["result"]["corpus"].get("sha256") == EXPECTED_SOURCE_SHA256[dataset]
                    for r in both)
            and all(str(r["result"]["corpus"].get("source_url", "")).startswith("r2://")
                    for r in both)
            and _same([r["binding"]["sha256"] for r in both])
            and _same([(r.get("runtime") or {}).get("native_vendor") for r in both])
            and all((r.get("runtime") or {}).get("native_vendor") == expected_vendor for r in both)
            and all((r.get("runtime") or {}).get("native_numeric_mode") == 1 for r in both)
            and _same([(r.get("runtime") or {}).get("source_sha256") for r in both]))
        ab_exact = bool(both) and _same([_trajectory(r) for r in both])
        bm = row["arms"]["baseline"]["outer_medians_seconds"]
        cm = row["arms"]["candidate"]["outer_medians_seconds"]
        ratio = (statistics.median(cm) / statistics.median(bm)
                 if bm and cm and statistics.median(bm) > 0 else float("inf"))
        row.update(provenance_ok=provenance_ok, ab_exact=ab_exact,
                   median_of_process_medians_ratio=ratio,
                   conservative_ratio_diagnostic=(max(cm) / min(bm) if bm and cm else float("inf")))
        row["pass"] = (row["arms"]["baseline"]["valid"] and
                       row["arms"]["candidate"]["valid"] and
                       provenance_ok and ab_exact and ratio < 1.0)
        if not provenance_ok:
            failures.append("%s provenance invalid" % dataset)
        if not ab_exact:
            failures.append("%s full training trajectory differs" % dataset)
        if ratio >= 1.0:
            failures.append("%s median ratio %.6f is not faster" % (dataset, ratio))
        rows.append(row)

    sabotage_ok = True
    for dataset in DATASETS:
        ss = [r for r in sabotage if r.get("dataset") == dataset]
        clean = sorted(cells.get(dataset, {}).get("candidate", []), key=lambda r: r.get("outer", -1))
        sab = ss[0] if len(ss) == 1 else None
        base = clean[0] if clean else None
        ok = (bool(sab and base)
              and ss[0].get("outer") == 0 and ss[0].get("launch_position") == 0
              and sab["result"].get("shape") == TARGET_SHAPE
              and sab["result"].get("parameters") == TARGET_PARAMETERS
              and sab["result"].get("steps_completed") == 1
              and sab["result"].get("limited") is False
              and sab.get("numeric_mode_env") == "identical"
              and sab["result"].get("attention_arm") == SWZ_ARM + "+sabotage_new"
              and sab["result"].get("attention_arm_resolved_hd64") == SWZ_ARM
              and sab["result"].get("attention_arm_trial_build") is True
              and _complete_witnesses(sab["result"], 1)
              and sab.get("commit") == base.get("commit")
              and sab.get("device_identity", {}).get("bytes", 0) > 0
              and sab.get("device_identity", {}).get("sha256") ==
                  base.get("device_identity", {}).get("sha256")
              and sab["result"].get("corpus") == base["result"].get("corpus")
              and sab["input"].get("initial_parameters_sha256") ==
                  base["input"].get("initial_parameters_sha256")
              and sab["input"].get("ids_sha256") == base["input"].get("ids_sha256", [])[:1]
              and sab["binding"]["sha256"] == base["binding"]["sha256"]
              and (sab.get("runtime") or {}).get("native_vendor") == expected_vendor
              and (sab.get("runtime") or {}).get("native_numeric_mode") == 1
              and (sab.get("runtime") or {}).get("source_sha256") ==
                  (base.get("runtime") or {}).get("source_sha256")
              # +sabotage_new is intentionally in the backward kernel: the
              # already-computed forward loss must stay fixed while every
              # downstream gradient/optimizer state proves branch reach.
              and sab["result"]["step_witnesses"][0]["sha256"]["loss"] ==
                  base["result"]["step_witnesses"][0]["sha256"]["loss"]
              and all(sab["result"]["step_witnesses"][0]["sha256"][name] !=
                      base["result"]["step_witnesses"][0]["sha256"][name]
                      for name in ("gradients", "parameters", "m", "v"))
              and sab["result"]["step_losses"][0] == base["result"]["step_losses"][0])
        sabotage_ok = sabotage_ok and ok
        if not ok:
            failures.append("%s candidate sabotage did not prove execution" % dataset)

    summary = {
        "pass": not failures and sabotage_ok and all(row["pass"] for row in rows),
        "provider": provider, "baseline_arm": BASE_ARM, "candidate_arm": SWZ_ARM,
        "sabotage_ok": sabotage_ok,
        "rule": ("full bit-identical training trajectory on exactly Taxi and Istella; concrete "
                 "native route plus sabotage; candidate median-of-process-medians below baseline "
                 "on both; stability and conservative extremes are diagnostics"),
        "cells": rows, "failures": failures,
    }
    with open(args.json, "w") as stream:
        json.dump(summary, stream, indent=1, sort_keys=True, allow_nan=False)
    for row in rows:
        print("ATTN BSWZ dataset=%s ratio=%.6f exact=%s provenance=%s pass=%s" %
              (row["dataset"], row["median_of_process_medians_ratio"], row["ab_exact"],
               row["provenance_ok"], row["pass"]))
    if failures:
        print("ATTN BSWZ REJECT " + "; ".join(failures))
        return 1
    print("ATTN BSWZ PROMOTE %s exact repeated training passed" % provider)
    return 0


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    rec = sub.add_parser("record")
    rec.add_argument("--dataset", choices=DATASETS, required=True)
    rec.add_argument("--arm", choices=ARMS + ("sabotage",), required=True)
    rec.add_argument("--outer", type=int, required=True)
    rec.add_argument("--launch-position", type=int, choices=(0, 1), required=True)
    rec.add_argument("--commit", required=True)
    rec.add_argument("--device-identity", required=True)
    rec.add_argument("--binding", required=True)
    rec.add_argument("--result", required=True)
    rec.add_argument("--events", required=True)
    rec.add_argument("--json", required=True)
    rec.set_defaults(func=cmd_record)
    summary = sub.add_parser("summarize")
    summary.add_argument("files", nargs="+")
    summary.add_argument("--provider", choices=("nvidia", "amd"), required=True)
    summary.add_argument("--json", required=True)
    summary.set_defaults(func=cmd_summarize)
    args = parser.parse_args()
    return args.func(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
