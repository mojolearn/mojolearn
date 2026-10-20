"""Read-only CPU-opponent coverage inventory. Does not run or admit measurements."""
import collections
import gzip
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
CHECKOUT = ROOT.parent / "checkout"
REPO = Path("/Users/andrewhendel/CascadeProjects/mojolearn")
sys.path.insert(0, str(CHECKOUT / "tools"))
import bench_board as B
import main_board_ingest as M
import race_fairness as RF

MODES = {"apple": "fast", "nvidia": "identical", "amd": "identical"}
TARGETS = {"apple": "m3 ultra", "nvidia": "l40s", "amd": "mi325x"}
expected = {}
plans = {}
excluded = []
settings = {}
configs = {}

def key(vendor, race, arm, phase, batch=None):
    return (vendor, race.get("family"), race.get("lane"), race.get("dataset"),
            race.get("rows"), race.get("shape") if isinstance(race.get("shape"), str) else None,
            arm, phase, batch)

for vendor, mode in MODES.items():
    plan = B.plan_races(vendor, [mode])
    plans[vendor] = plan
    for race in plan:
        settings[(vendor, race["id"])] = B.race_settings({"vendor": vendor, "rounds": 1}, race)
        configs[(vendor, race["id"])] = M.current_lane_config(B, race["family"], race["lane"])
        for arm in race["opponents"]:
            if B.arm_device(arm, vendor) != "cpu":
                continue
            reason = B._planned_refusal(vendor, B.smoke_key(race), arm)
            reason = reason or RF.exclusions(dict(lane=race["lane"], cells=[
                dict(arm=arm, mode="opponent")])).get(arm)
            # Both CPU and CUDA tree arms refuse unconditionally in
            # speed_gbdt_arm.lightgbm_arms / lightgbm_rank_arms. The smoke
            # PLANNED_REFUSALS table currently names only the CUDA arm.
            if race["family"] == "trees" and arm == "lightgbm-cpu":
                reason = ("driver refuses this tree arm before timing: "
                          "MIN-CHILD / ROW-SAMPLING / CATEGORICAL-ALGORITHM / DIFFERENT-ALGORITHM")
            batches = [b for a, b in B.INFER.plan_cells(dict(race, arms=[arm])) if a == arm]
            if race["family"] == "algos" and B.ALGOS.has_infer(race["lane"]):
                batches = ["Xq"]
            for phase, batch in [("fit", None)] + [("infer", b) for b in batches]:
                k = key(vendor, race, arm, phase, batch)
                row = dict(vendor=vendor, mode=mode, race_id=race["id"], family=race["family"],
                           lane=race["lane"], dataset=race["dataset"], arm=arm,
                           phase=phase, batch=batch)
                if reason:
                    excluded.append(dict(row, reason=reason))
                else:
                    expected[k] = row

candidates = collections.defaultdict(list)
scan_counts = collections.Counter()
errors = []
seen = set()

def identify(box, path, explicit_vendor=None, machine=None):
    gpu = box.get("gpu") or {}
    vendor = explicit_vendor or gpu.get("vendor")
    if vendor == "metal":
        vendor = "apple"
    if vendor not in MODES:
        return None, "unknown"
    model = str(machine or gpu.get("name") or "").lower()
    if TARGETS[vendor] in model:
        return vendor, "matching_model"
    if model and model not in ("none", "-", "unknown"):
        return vendor, "other_model"
    return vendor, "unknown"

def record(vendor, hw_status, race, cell, phase, path, entry=None, store_key=None, box=None):
    if vendor not in MODES or not isinstance(cell, dict):
        return
    if cell.get("device") != "cpu" or str(cell.get("arm", "")).startswith("ours"):
        return
    k = key(vendor, race, cell.get("arm"), phase, cell.get("batch") if phase == "infer" else None)
    if k not in expected:
        scan_counts["out_of_current_plan_cells"] += 1
        return
    if hw_status == "other_model":
        scan_counts["other_hardware_cells"] += 1
        return
    timing = cell.get("median_ms")
    ok = cell.get("status") in ("ok", "not-scored") and isinstance(timing, (int, float)) and math.isfinite(timing) and timing >= 0
    original_settings = cell.get("settings")
    if isinstance(original_settings, str):
        try:
            original_settings = json.loads(original_settings)
        except ValueError:
            original_settings = None
    current = settings[(vendor, expected[k]["race_id"])]
    if store_key and store_key.get("settings_sha256"):
        exact_settings = store_key["settings_sha256"] == B.STORE.sha256_json(current)
    else:
        exact_settings = original_settings == current if original_settings else None
    current_cfg = configs[(vendor, expected[k]["race_id"])]
    old_cfg = M.recorded_lane_config(race)
    if old_cfg is None and isinstance(original_settings, dict):
        old_cfg = original_settings.get("lane_config")
    config_match = (M._settings_view(current_cfg, race["family"]) == M._settings_view(old_cfg, race["family"])
                    if current_cfg is not None and old_cfg is not None else None)
    audit = (entry or {}).get("opponents", {}).get(cell.get("arm"), {})
    admitted = bool(entry and cell.get("arm") in entry.get("admitted_opponents", [])
                    and not audit.get("rerace"))
    measurement = cell.get("measurement") or {}
    params = cell.get("params") or measurement.get("parameters") or {}
    params = params or ((race.get("params") or {}).get("arms") or {}).get(cell.get("arm"), {})
    fingerprint = (k, cell.get("hash"), timing, cell.get("status"),
                   B.STORE.sha256_json(params), B.STORE.sha256_json(original_settings),
                   admitted, config_match, hw_status)
    if fingerprint in seen:
        scan_counts["duplicate_cells"] += 1
        return
    seen.add(fingerprint)
    summary = dict(path=str(path), status=cell.get("status"), median_ms=timing, successful=ok,
                   hardware=hw_status, audit_admitted=admitted, audit_verdict=audit.get("verdict"),
                   audit_reason=audit.get("reason"), exact_settings_match=exact_settings,
                   lane_config_match=config_match, parameters_present=bool(params),
                   measurement_completeness=measurement.get("completeness"),
                   measured_at=(store_key or {}).get("measured_at") or race.get("finished"),
                   library_version=cell.get("library_version") or (store_key or {}).get("library_version"))
    candidates[k].append(summary)

def read_board(path):
    doc = json.loads(path.read_text())
    box = doc.get("box") or {}
    vendor, hardware = identify(box, path, (doc.get("config") or {}).get("vendor"))
    if vendor is None or hardware == "other_model":
        scan_counts["boards_unknown_or_other_hardware"] += 1
        return
    # Recompute audit decisions in memory using the existing board tool.
    # This does not write a board or create a new measurement.
    RF.apply(doc)
    for section in ("races", "extra_races"):
        for race in (doc.get(section) or {}).values():
            for phase, field in (("fit", "cells"), ("infer", "infer_cells")):
                entry = (race.get("audit") or {}).get(MODES[vendor] + "/" + phase)
                for cell in race.get(field) or []:
                    record(vendor, hardware, race, cell, phase, path, entry=entry, box=box)

def read_store(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                sk = rec.get("key") or {}
                if not sk:
                    continue
                vendor, hw = identify({}, path, sk.get("vendor"), sk.get("machine"))
                race = {n: sk.get(n) for n in ("family", "lane", "dataset", "rows")}
                race["shape"] = sk.get("neural_shape")
                race["finished"] = rec.get("measured_at")
                record(vendor, hw, race, rec.get("cell") or {}, "fit", f"{path}:{number}", store_key=sk)
                for cell in rec.get("infer_cells") or []:
                    record(vendor, hw, race, cell, "infer", f"{path}:{number}", store_key=sk)
            except Exception as exc:
                errors.append(dict(path=str(path), line=number, error=repr(exc)))

def read_archive_cells(path):
    vendor = {"m3-ultra": "apple", "nvidia-l40s": "nvidia", "amd-mi325x": "amd"}.get(path.parent.name)
    if vendor is None:
        return
    with gzip.open(path, "rt") as stream:
        for number, line in enumerate(stream, 1):
            cell = json.loads(line)
            race = {n: cell.get(n) for n in ("family", "lane", "dataset", "rows")}
            race["shape"] = cell.get("neural_shape")
            phase = "infer" if str(cell.get("_source", "")).endswith(("infer", ":infer")) else "fit"
            # The archive grouping locates candidates; it is not worker hardware
            # evidence. Without the source board, leave hardware unresolved.
            record(vendor, "unknown", race, cell, phase, f"{path}:{number}")

paths = [Path(p) for p in (ROOT / "sources.list").read_text().splitlines()]
paths += list((REPO / "bench/results/opponent-archive").glob("*/opponent-store.jsonl.gz"))
paths += list((REPO / "bench/results/opponent-archive").glob("*/opponent-cells-all.jsonl.gz"))
# Canonical boards first so duplicate retention prefers their reviewable paths.
paths.sort(key=lambda p: ("/bench/results/bench_board/" not in str(p), str(p)))
for path in paths:
    try:
        if path.name == "board.json":
            read_board(path)
        elif path.name == "opponent-cells-all.jsonl.gz":
            read_archive_cells(path)
        else:
            read_store(path)
        scan_counts["files_read"] += 1
    except Exception as exc:
        errors.append(dict(path=str(path), error=repr(exc)))

rows = []
counts = collections.defaultdict(collections.Counter)
for k, row in expected.items():
    cs = candidates[k]
    success = [c for c in cs if c["successful"]]
    admitted = [c for c in success if c["audit_admitted"] and c["hardware"] == "matching_model"
                and c["lane_config_match"] is not False]
    if admitted:
        category = "retained_admitted_candidate"
    elif success:
        category = "successful_history_needs_review"
    elif cs:
        category = "only_failed_or_incomplete_history"
    else:
        category = "no_retained_candidate_found"
    row = dict(row, category=category, candidates=cs)
    rows.append(row)
    counts[row["vendor"] + "/" + row["phase"]][category] += 1

summary = dict(scope="Full current Apple fast (M3 Ultra), NVIDIA identical (L40S), AMD identical (MI325X). "
                     "Fit and prediction batches counted separately. No new worker readbacks or races. "
                     "A retained admitted candidate is not automatic approval against an unmeasured new own arm.",
               source_commit="3141d7bdc8936971655a7732c709a91e47925f20",
               counts=dict(counts), excluded_counts=dict(collections.Counter(
                   r["vendor"] + "/" + r["phase"] for r in excluded)),
               scanned=dict(scan_counts), errors=errors)
(ROOT / "coverage.json").write_text(json.dumps(dict(summary=summary, rows=rows, excluded=excluded), indent=2) + "\n")
(ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
