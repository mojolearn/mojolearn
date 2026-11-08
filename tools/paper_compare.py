#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn
"""tools/paper_compare.py -- rerun ONLY the measurements of the mlsys paper on a newer published release and
compare (Andrew, 2026-11-08: "rerun JUST the measurements we have in the mlsys on our 0.8.66 to see how we compare
and if we have corrected regressions").

The paper's boards (0.8.55, NVIDIA L40S and AMD MI300X; the merged per-vendor boards under
~/mojolearn-evidence/mlsys27-evidence/measurements/paper-tables/, READ ONLY: that repo is under review and is never
written) hold every race the paper reports: the classical and tree lanes with a GPU-library opponent and the neural
board. This tool:

  recipes  --vendor V [--board PAPER_BOARD] [--out JSON]
        extracts the exact recipe set of one vendor's paper board: one recipe per OUR arm cell (fit, and each
        inference batch), with its race id, family, lane, dataset, shape, numeric mode, the explicit parameters the
        board read back (the paper copy is anonymized; the library name is restored), and the 0.8.55 numbers
        (median_ms, the kernel span where the board derived one, quality, status) plus every paired opponent's
        stored time and ratio, exactly as the paper's audit recorded them. Opponents are NEVER rerun (Andrew: opponents
        are scored once); the compare reuses these numbers.
        Default out: bench/results/paper_compare/recipes-<vendor>.json.
  ids      --recipes JSON --planner tools/bench_board.py --out-dir DIR
        ON THE BOX: the recipe races the current board plans (bench_board.plan_races with the recipe file's
        bench_board_args), written as race-ids-full.txt and race-ids-smoke.txt (the smoke's rows and shape), the
        board arguments as board-args.txt (one per line) and the races the board no longer plans as
        not-planned.json. A race the board does not plan is recorded, never a refusal of the whole run.
  pass-ids --full-ids FILE --smoke-json SMOKE.json... --out FILE
        the full-size ids of the planned races whose smoke passed (smoke-failed.json beside --out lists the rest).
  compare  --recipes JSON --board BOARD.json... [--out-md MD] [--out-json JSON] [--summary TXT]
        0.8.55 against the new board, per recipe: ms both, ratio, quality both, the flags below, and the ratio
        against the paper's stored opponent before and after. Prints a one-screen summary.

FLAGS (per recipe; a recipe can carry several)
  REGRESSION     new / old median_ms > 1.10 (slower)          FASTER   new / old < 0.90
  SAME           0.90 <= ratio <= 1.10
  FIXED          0.8.55 had no time (refused, failed) and the new release has one
  BROKEN         0.8.55 had a time and the new release has none (its status is shown)
  STILL-FAILING  neither release has a time
  NOT-RUN        no new cell: the board no longer plans the race, its smoke failed, or the job did not reach it
  QUALITY-DROP   tools/board_quality.compare(new, old) FAILs on a one-sided score or a reference bound: the new
                 release is worse than 0.8.55 beyond the board's declared margin (the quality-parity policy)
  QUALITY-CHANGE the same policy FAILs on a symmetric metric (no better side)
  NOW-WINS       slower than the paper's fastest stored opponent in 0.8.55, at least as fast now
  NOW-LOSES      the reverse
  PARAMS-DIFF    the explicit parameters our arm read back differ from the paper's
  HW-DIFF        the new board's GPU is not the paper's GPU (timings are then not comparable)

The new numbers come from tools/ops/paper_compare_job.sh, which runs these recipes through tools/bench_board.py
(our arms only) from the published wheels. Tests: tools/test_paper_compare.py.
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import board_quality as BQ  # noqa: E402

SCHEMA = "mojolearn-paper-compare-recipes/1"
COMPARE_SCHEMA = "mojolearn-paper-compare/1"
PAPER_VERSION = "0.8.55"
PAPER_ROOT = os.path.expanduser("~/mojolearn-evidence/mlsys27-evidence/measurements/paper-tables")
OUT_DIR = os.path.join(REPO, "bench", "results", "paper_compare")
SLOWER = 1.10        # new / old above this: REGRESSION (brief, 2026-11-08)
FASTER = 0.90        # new / old below this: FASTER
VENDORS = ("nvidia", "amd")
# The paper copy is anonymized (the supplement names no library). Restore the name in what we compare.
ANON = (("ANONLIB", "MOJOLEARN"), ("Anonlib", "Mojolearn"), ("anonlib", "mojolearn"))
# Statuses with no usable time. "not-scored" keeps its time: the scorer failed, the clock did not.
NO_TIME = re.compile(r"^(REFUSED|FAILED|UNKNOWN|ERROR|TIMEOUT|SKIPPED|CRASHED)", re.I)
SMOKE_SHAPE = "small"


# ---------------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------------

def load_json(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        return json.load(fh)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def deanon(value):
    """The value with the anonymized library name restored, recursively (keys too)."""
    if isinstance(value, str):
        for a, b in ANON:
            value = value.replace(a, b)
        return value
    if isinstance(value, list):
        return [deanon(v) for v in value]
    if isinstance(value, dict):
        return {deanon(k): deanon(v) for k, v in value.items()}
    return value


def number(value):
    return BQ.number(value)


def timed(status, ms):
    return number(ms) and not NO_TIME.match(str(status or ""))


def short_status(status):
    s = str(status if status is not None else "missing")
    return s if len(s) <= 300 else s[:297] + "..."


def recipe_key(race_id, phase, arm, batch):
    return " ".join(x for x in (race_id, phase, arm, batch or "") if x)


def scored_quality(quality):
    """Only the metrics the quality-parity policy declares (the ones the compare judges)."""
    q = quality or {}
    keep = set(BQ.ABSOLUTE) | set(BQ.RELATIVE) | set(BQ.REFERENCE) | set(BQ.AGREEMENT)
    return {k: v for k, v in q.items() if k in keep}


def kernel_ms(cell):
    span = ((cell.get("comparability") or {}).get("span") or {})
    v = span.get("fit_minus_upload_ms")
    return v if number(v) else None


def gpu_name(board):
    gpu = ((board.get("box") or {}).get("gpu") or {})
    return gpu.get("name")


# ---------------------------------------------------------------------------------------------------------------
# recipes
# ---------------------------------------------------------------------------------------------------------------

def _opponents_for(race, phase, mode, arm, batch):
    """The paper's stored opponent timings paired with this arm and batch (the audit's preferred clocks)."""
    audit = (race.get("audit") or {}).get("%s/%s" % (mode, phase)) or {}
    cells = race.get("cells" if phase == "fit" else "infer_cells") or []
    out = []
    for opp, od in sorted((audit.get("opponents") or {}).items()):
        if (od.get("ours_arm") or "ours") != arm:
            continue
        tcs = od.get("timing_comparisons") or ([od["timing_comparison"]] if od.get("timing_comparison") else [])
        for tc in tcs:
            if (tc.get("batch") or None) != (batch or None):
                continue
            ocell = next((c for c in cells if c.get("arm") == opp and (c.get("batch") or None) == (batch or None)),
                         {})
            out.append({
                "arm": opp,
                "opponent_ms": tc.get("opponent_ms") if number(tc.get("opponent_ms")) else None,
                "ours_ms": tc.get("ours_ms") if number(tc.get("ours_ms")) else None,
                "clock": tc.get("clock"),
                "label": tc.get("label"),
                "ratio": tc.get("value") if number(tc.get("value")) else None,
                "verdict": od.get("verdict"),
                "opponent_status": short_status(ocell.get("status")) if ocell else None,
            })
    return out


def extract_recipes(board, vendor, source_path=None, version=PAPER_VERSION):
    """The recipe document of one paper board (see the module docstring)."""
    races = board.get("races") or {}
    recipes, race_ids = [], []
    families, lanes, datasets = [], [], []
    for rid in sorted(races):
        race = races[rid]
        ours = race.get("our_arms") or {}
        if not ours:
            continue
        race_ids.append(rid)
        fam, lane, ds = race.get("family"), race.get("lane"), race.get("dataset")
        for lst, v in ((families, fam), (lanes, lane)):
            if v not in lst:
                lst.append(v)
        if fam != "neural" and ds not in datasets:
            datasets.append(ds)
        for phase, cells in (("fit", race.get("cells") or []), ("infer", race.get("infer_cells") or [])):
            for cell in cells:
                arm = cell.get("arm")
                if arm not in ours:
                    continue
                batch = cell.get("batch")
                mode = ours[arm]
                params = deanon((cell.get("params") or {}).get("params"))
                impl = deanon((cell.get("params") or {}).get("implementation"))
                recipes.append({
                    "key": recipe_key(rid, phase, arm, batch),
                    "race_id": rid, "family": fam, "lane": lane, "dataset": ds,
                    "rows": race.get("rows"), "shape": cell.get("shape"),
                    "neural_shape": cell.get("neural_shape"),
                    "phase": phase, "arm": arm, "mode": mode,
                    "batch": batch, "batch_rows": cell.get("batch_rows"),
                    "implementation": impl, "params": params,
                    "call": deanon(cell.get("call")),
                    "params_check": race.get("params_check"),
                    "v0855": {
                        "status": short_status(cell.get("status")),
                        "median_ms": cell.get("median_ms") if number(cell.get("median_ms")) else None,
                        "times_ms": cell.get("times_ms"),
                        "kernel_ms": kernel_ms(cell) if phase == "fit" else None,
                        "quality": cell.get("quality") or {},
                        "hash": cell.get("hash"),
                        "device_name": cell.get("device_name"),
                        "library_version": cell.get("library_version"),
                    },
                    "opponents": _opponents_for(race, phase, mode, arm, batch),
                })
    order = ("trees", "classical", "classical2", "neural", "algos")
    families.sort(key=lambda f: order.index(f) if f in order else len(order))
    doc = {
        "schema": SCHEMA,
        "vendor": vendor,
        "paper_version": version,
        "source": {
            "board": source_path,
            "sha256": sha256_file(source_path) if source_path and os.path.isfile(source_path) else None,
            "gpu": gpu_name(board),
            "created": board.get("created"), "updated": board.get("updated"),
            "note": "the paper's merged board (anonymized copy; library name restored in params and calls)",
        },
        # what tools/ops/paper_compare_job.sh passes to tools/bench_board.py: our arms only (no opponent is
        # raced: the default of the board without --with-opponents), IDENTICAL, one round, GPU opponents only
        # (keeps exactly the own pairing arms of GPU opponents, which is what the paper raced)
        "bench_board_args": {
            "vendor": vendor, "families": families, "lanes": lanes, "datasets": datasets,
            "modes": "identical", "rounds": 1, "gpu_opponents_only": True,
        },
        "thresholds": {"regression_ratio_above": SLOWER, "faster_ratio_below": FASTER,
                       "quality_policy": BQ.POLICY},
        "race_ids": race_ids,
        "counts": {"races": len(race_ids), "recipes": len(recipes),
                   "fit": sum(r["phase"] == "fit" for r in recipes),
                   "infer": sum(r["phase"] == "infer" for r in recipes),
                   "by_family": {f: sum(r["family"] == f for r in recipes) for f in families}},
        "recipes": recipes,
    }
    return doc


def cmd_recipes(args):
    board_path = args.board or os.path.join(PAPER_ROOT, "board-%s.json.gz" % args.vendor)
    doc = extract_recipes(load_json(board_path), args.vendor, board_path, args.version)
    out = args.out or os.path.join(OUT_DIR, "recipes-%s.json" % args.vendor)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
        fh.write("\n")
    c = doc["counts"]
    print("PAPER-RECIPES vendor=%s races=%d recipes=%d fit=%d infer=%d gpu=%s out=%s"
          % (args.vendor, c["races"], c["recipes"], c["fit"], c["infer"], doc["source"]["gpu"], out))
    return 0


# ---------------------------------------------------------------------------------------------------------------
# ids (on the box)
# ---------------------------------------------------------------------------------------------------------------

def load_planner(path):
    spec = importlib.util.spec_from_file_location("paper_compare_planner", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    spec.loader.exec_module(mod)
    return mod


def board_args(bargs):
    """tools/bench_board.py arguments for the recipe set, one list."""
    out = ["--vendor", bargs["vendor"], "--families", ",".join(bargs["families"]),
           "--lanes", ",".join(bargs["lanes"]), "--modes", bargs.get("modes") or "identical"]
    if bargs.get("datasets"):
        out += ["--datasets", ",".join(bargs["datasets"])]
    if bargs.get("gpu_opponents_only"):
        out.append("--gpu-opponents-only")
    return out


def smoke_id(rid, smoke_rows):
    rid = re.sub(r"/rows=full$", "/rows=%d" % int(smoke_rows), rid)
    return re.sub(r"/shape=full$", "/shape=" + SMOKE_SHAPE, rid)


def full_id(rid):
    rid = re.sub(r"/rows=[0-9]+$", "/rows=full", rid)
    return re.sub(r"/shape=%s$" % SMOKE_SHAPE, "/shape=full", rid)


def plan_ids(doc, planner):
    """(full ids planned, smoke ids planned, {race id: why not planned})."""
    b = doc["bench_board_args"]
    vendor = b["vendor"]
    modes = planner.modes_for(vendor, b.get("modes"))
    families = tuple(b["families"])
    lanes = tuple(b["lanes"])
    datasets = tuple(b.get("datasets") or ())
    known = set()
    for f in families:
        known |= set(planner.family_lanes(f))
    lanes_ok = tuple(l for l in lanes if l in known)

    def plan(rows, shape):
        races = planner.plan_races(vendor, modes, families, lanes_ok, datasets or planner.DATASETS, rows, shape)
        if b.get("gpu_opponents_only"):
            races = planner.gpu_opponents_only(races, vendor)
        return {r["id"]: r for r in races}

    full = plan(None, "full")
    smoke_rows = getattr(planner, "SMOKE_ROWS", 2000)
    smoke = plan(int(smoke_rows), SMOKE_SHAPE)
    wanted_arms = {}
    for r in doc["recipes"]:
        wanted_arms.setdefault(r["race_id"], set()).add(r["arm"])
    planned_full, planned_smoke, missing = [], [], {}
    for rid in doc["race_ids"]:
        race = full.get(rid)
        if race is None:
            lane = rid.split("/")[1] if "/" in rid else rid
            missing[rid] = ("lane %s is no longer a board lane" % lane if lane not in known
                            else "the board no longer plans this race on %s" % vendor)
            continue
        sid = smoke_id(rid, smoke_rows)
        if sid not in smoke:
            missing[rid] = "the board plans no smoke race %s" % sid
            continue
        lost = sorted(wanted_arms.get(rid, set()) - set(race.get("our_arms") or {}))
        if lost:
            # recorded per arm by the compare (NOT-RUN); the race still runs for the arms it plans
            missing[rid + " arms"] = "our arm(s) %s not planned" % ",".join(lost)
        planned_full.append(rid)
        planned_smoke.append(sid)
    return planned_full, planned_smoke, missing


def cmd_ids(args):
    doc = load_json(args.recipes)
    planner = load_planner(args.planner)
    full, smoke, missing = plan_ids(doc, planner)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "race-ids-full.txt"), "w") as fh:
        fh.write("".join(x + "\n" for x in full))
    with open(os.path.join(args.out_dir, "race-ids-smoke.txt"), "w") as fh:
        fh.write("".join(x + "\n" for x in smoke))
    with open(os.path.join(args.out_dir, "board-args.txt"), "w") as fh:
        fh.write("".join(x + "\n" for x in board_args(doc["bench_board_args"])))
    with open(os.path.join(args.out_dir, "not-planned.json"), "w") as fh:
        json.dump(missing, fh, indent=1, sort_keys=True)
    print("PAPER-IDS vendor=%s recipe_races=%d planned=%d not_planned=%d"
          % (doc["vendor"], len(doc["race_ids"]), len(full), len([k for k in missing if not k.endswith(" arms")])))
    return 0


def passed_full_ids(full_ids, smoke_docs):
    """(full ids whose smoke passed, {full id: why not})."""
    passed, seen = set(), {}
    for doc in smoke_docs:
        for key, v in (doc.get("races") or {}).items():
            fid = full_id(v.get("id") or key)
            seen[fid] = v
            if v.get("pass") is True:
                passed.add(fid)
    keep, failed = [], {}
    for rid in full_ids:
        if rid in passed:
            keep.append(rid)
        elif rid in seen:
            v = seen[rid]
            fails = v.get("failures") or []
            failed[rid] = short_status("smoke failed: " + "; ".join(
                "%s: %s" % (f.get("arm"), f.get("reason")) for f in fails if isinstance(f, dict)) if fails
                else "smoke failed")
        else:
            failed[rid] = "no smoke result"
    return keep, failed


def cmd_pass_ids(args):
    with open(args.full_ids) as fh:
        full = [x.strip() for x in fh if x.strip()]
    docs = []
    for p in args.smoke_json:
        try:
            docs.append(load_json(p))
        except (OSError, ValueError) as exc:
            print("PAPER-PASS-IDS unreadable smoke %s: %s" % (p, exc), file=sys.stderr)
    keep, failed = passed_full_ids(full, docs)
    with open(args.out, "w") as fh:
        fh.write("".join(x + "\n" for x in keep))
    with open(os.path.join(os.path.dirname(os.path.abspath(args.out)), "smoke-failed.json"), "w") as fh:
        json.dump(failed, fh, indent=1, sort_keys=True)
    print("PAPER-PASS-IDS planned=%d smoke_passed=%d smoke_failed=%d" % (len(full), len(keep), len(failed)))
    return 0


# ---------------------------------------------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------------------------------------------

def merge_boards(boards):
    races, gpus, versions = {}, [], []
    for b in boards:
        g = gpu_name(b)
        if g and g not in gpus:
            gpus.append(g)
        for rid, race in (b.get("races") or {}).items():
            races[rid] = race
            for c in race.get("cells") or []:
                v = c.get("library_version")
                if str(c.get("arm", "")).startswith("ours") and v and v not in versions:
                    versions.append(v)
    return races, gpus, versions


def find_cell(race, phase, arm, batch):
    cells = race.get("cells" if phase == "fit" else "infer_cells") or []
    return next((c for c in cells if c.get("arm") == arm and (c.get("batch") or None) == (batch or None)), None)


def params_diff(old, new):
    if old is None or new is None:
        return []
    keys = sorted(set(old) | set(new))
    return [k for k in keys if old.get(k) != new.get(k)]


def same_clock_ms(opp, cell, phase):
    """Our new time on the opponent comparison's clock: whole = median_ms, kernel = the derived kernel span."""
    if cell is None:
        return None
    if opp.get("clock") == "kernel":
        return kernel_ms(cell) if phase == "fit" else None
    if opp.get("clock") == "whole":
        return cell.get("median_ms") if number(cell.get("median_ms")) else None
    return None       # MIXED clocks: the paper shows no ratio; neither do we


def compare_one(rec, races, not_run_why, hw_diff):
    old = rec["v0855"]
    race = races.get(rec["race_id"])
    cell = find_cell(race, rec["phase"], rec["arm"], rec["batch"]) if race else None
    old_t = timed(old["status"], old["median_ms"])
    row = {"key": rec["key"], "race_id": rec["race_id"], "family": rec["family"], "lane": rec["lane"],
           "dataset": rec["dataset"], "phase": rec["phase"], "arm": rec["arm"], "batch": rec["batch"],
           "old_ms": old["median_ms"] if old_t else None, "old_status": old["status"],
           "old_quality": scored_quality(old["quality"]), "flags": []}
    if cell is None:
        row.update(new_ms=None, new_status="missing", ratio=None, new_quality={},
                   why=not_run_why.get(rec["race_id"]) or ("race not in the new board" if race is None
                                                          else "no %s cell for arm %s" % (rec["phase"], rec["arm"])))
        row["flags"].append("NOT-RUN")
        row["opponents"] = [dict(o, ratio_new=None) for o in rec["opponents"]]
        return row
    new_status = short_status(cell.get("status"))
    new_t = timed(new_status, cell.get("median_ms"))
    row.update(new_ms=cell.get("median_ms") if new_t else None, new_status=new_status,
               new_quality=scored_quality(cell.get("quality")))
    if old_t and new_t:
        row["ratio"] = row["new_ms"] / row["old_ms"] if row["old_ms"] else None
        r = row["ratio"]
        row["flags"].append("REGRESSION" if r is not None and r > SLOWER else
                            "FASTER" if r is not None and r < FASTER else "SAME")
    else:
        row["ratio"] = None
        row["flags"].append("FIXED" if new_t else "BROKEN" if old_t else "STILL-FAILING")
    # quality: the new release judged against 0.8.55 under the board's own policy (new = "ours")
    q = BQ.compare({"arm": rec["arm"], "lane": rec["lane"], "quality": cell.get("quality") or {},
                    "status": cell.get("status")},
                   {"arm": rec["arm"], "lane": rec["lane"], "quality": old["quality"] or {},
                    "status": old["status"]})
    row["quality_verdict"] = q["verdict"]
    bad = [c for c in q["checks"] if c.get("verdict") == "FAIL"]
    row["quality_failures"] = ["%s %s->%s (%s, margin %s)" % (c["metric"], c.get("opponent"), c.get("ours"),
                                                            c.get("comparison"), c.get("margin")) for c in bad]
    if any(str(c.get("comparison", "")).startswith(("one-sided", "both-")) for c in bad) or q["failures"]:
        row["flags"].append("QUALITY-DROP")
    elif bad:
        row["flags"].append("QUALITY-CHANGE")
    # opponents: the paper's stored numbers, our new time on the same clock
    opps = []
    for o in rec["opponents"]:
        mine = same_clock_ms(o, cell, rec["phase"]) if new_t else None
        ratio_new = mine / o["opponent_ms"] if number(mine) and number(o.get("opponent_ms")) and o["opponent_ms"] \
            else None
        opps.append(dict(o, ours_ms_new=mine, ratio_new=ratio_new))
    row["opponents"] = opps
    best = [o for o in opps if number(o.get("opponent_ms")) and number(o.get("ratio"))]
    if best:
        b = min(best, key=lambda o: o["opponent_ms"])
        row["best_opponent"] = b
        if number(b.get("ratio_new")):
            if b["ratio"] > 1.0 >= b["ratio_new"]:
                row["flags"].append("NOW-WINS")
            elif b["ratio"] <= 1.0 < b["ratio_new"]:
                row["flags"].append("NOW-LOSES")
    new_params = (cell.get("params") or {}).get("params")
    diff = params_diff(rec["params"], new_params)
    row["params_diff"] = diff
    if diff:
        row["flags"].append("PARAMS-DIFF")
    if hw_diff:
        row["flags"].append("HW-DIFF")
    return row


def compare(doc, boards, not_run_why=None):
    races, gpus, versions = merge_boards(boards)
    paper_gpu = doc["source"].get("gpu")
    hw_diff = bool(gpus) and paper_gpu is not None and any(g != paper_gpu for g in gpus)
    rows = [compare_one(rec, races, not_run_why or {}, hw_diff) for rec in doc["recipes"]]
    counts = {}
    for r in rows:
        for f in r["flags"]:
            counts[f] = counts.get(f, 0) + 1
    return {"schema": COMPARE_SCHEMA, "vendor": doc["vendor"], "paper_version": doc["paper_version"],
            "new_versions": versions, "paper_gpu": paper_gpu, "new_gpus": gpus, "hw_diff": hw_diff,
            "thresholds": doc.get("thresholds"), "recipes": len(rows), "flag_counts": counts, "rows": rows}


def fmt_ms(v):
    if not number(v):
        return "-"
    return "%.0f" % v if v >= 100 else "%.2f" % v if v >= 1 else "%.3f" % v


def fmt_ratio(v):
    return "%.2f" % v if number(v) else "-"


def fmt_q(q):
    if not q:
        return "-"
    return " ".join("%s=%.4g" % (k, v) for k, v in sorted(q.items())[:3] if number(v)) or "-"


def render_md(res):
    v, new = res["vendor"], ", ".join(res["new_versions"]) or "?"
    lines = ["# Paper measurements: %s against %s on %s" % (res["paper_version"], new, v), ""]
    lines.append("Paper GPU: %s. New GPU: %s.%s" % (res["paper_gpu"], ", ".join(res["new_gpus"]) or "?",
                 " **Different hardware: timings are not comparable.**" if res["hw_diff"] else ""))
    t = res.get("thresholds") or {}
    lines.append("Ratio = new / %s median ms; REGRESSION above %s, FASTER below %s; quality judged by %s. "
                 "Opponents are the paper's stored cells (never rerun); 'vs opp' is ours over the paper's fastest "
                 "paired opponent on that comparison's clock." % (res["paper_version"],
                 t.get("regression_ratio_above"), t.get("faster_ratio_below"), t.get("quality_policy")))
    lines.append("")
    lines.append("Flags: " + ", ".join("%s %d" % kv for kv in sorted(res["flag_counts"].items())))
    lines.append("")
    lines.append("| recipe | %s ms | new ms | ratio | flags | quality %s | quality new | q | best opponent "
                 "(ms) | vs opp %s | vs opp new | note |" % ((res["paper_version"],) * 3))
    lines.append("|---|---:|---:|---:|---|---|---|---|---|---:|---:|---|")
    for r in res["rows"]:
        b = r.get("best_opponent") or {}
        note = []
        if r.get("why"):
            note.append(r["why"])
        if r.get("new_status") not in (None, "ok", "missing") and r.get("new_ms") is None:
            note.append("new: " + r["new_status"][:120])
        if r.get("old_ms") is None and r.get("old_status") != "ok":
            note.append("old: " + str(r["old_status"])[:120])
        if r.get("quality_failures"):
            note.append("quality: " + "; ".join(r["quality_failures"])[:200])
        if r.get("params_diff"):
            note.append("params: " + ",".join(r["params_diff"][:6]))
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["key"], fmt_ms(r.get("old_ms")), fmt_ms(r.get("new_ms")), fmt_ratio(r.get("ratio")),
            " ".join(r["flags"]), fmt_q(r.get("old_quality")), fmt_q(r.get("new_quality")),
            r.get("quality_verdict", "-"),
            "%s (%s)" % (b.get("arm"), fmt_ms(b.get("opponent_ms"))) if b else "-",
            fmt_ratio(b.get("ratio")), fmt_ratio(b.get("ratio_new")), "; ".join(note).replace("|", "/")))
    lines.append("")
    return "\n".join(lines)


def render_summary(res, top=8):
    rows = res["rows"]
    out = ["PAPER-COMPARE vendor=%s %s->%s recipes=%d gpu %s -> %s%s" % (
        res["vendor"], res["paper_version"], ",".join(res["new_versions"]) or "?", res["recipes"],
        res["paper_gpu"], ",".join(res["new_gpus"]) or "?", " HW-DIFF" if res["hw_diff"] else "")]
    out.append("flags: " + " ".join("%s=%d" % kv for kv in sorted(res["flag_counts"].items())))
    timed_rows = [r for r in rows if number(r.get("ratio"))]
    if timed_rows:
        rs = sorted(r["ratio"] for r in timed_rows)
        geo = 1.0
        for x in rs:
            geo *= x
        out.append("timed both=%d median ratio=%.2f geomean=%.2f" % (len(rs), rs[len(rs) // 2],
                                                                      geo ** (1.0 / len(rs))))
    worst = sorted([r for r in timed_rows if "REGRESSION" in r["flags"]], key=lambda r: -r["ratio"])[:top]
    if worst:
        out.append("worst regressions:")
        out += ["  %.2fx %s (%s -> %s ms)" % (r["ratio"], r["key"], fmt_ms(r["old_ms"]), fmt_ms(r["new_ms"]))
                for r in worst]
    best = sorted([r for r in timed_rows if "FASTER" in r["flags"]], key=lambda r: r["ratio"])[:top]
    if best:
        out.append("biggest gains:")
        out += ["  %.2fx %s (%s -> %s ms)" % (r["ratio"], r["key"], fmt_ms(r["old_ms"]), fmt_ms(r["new_ms"]))
                for r in best]
    for flag in ("FIXED", "BROKEN", "QUALITY-DROP", "QUALITY-CHANGE", "NOW-WINS", "NOW-LOSES", "PARAMS-DIFF",
                 "NOT-RUN"):
        hit = [r for r in rows if flag in r["flags"]]
        if hit:
            out.append("%s (%d): %s%s" % (flag, len(hit), ", ".join(r["key"] for r in hit[:6]),
                                          " ..." if len(hit) > 6 else ""))
    return "\n".join(out)


def cmd_compare(args):
    doc = load_json(args.recipes)
    boards = []
    for p in args.board:
        try:
            boards.append(load_json(p))
        except (OSError, ValueError) as exc:
            print("PAPER-COMPARE unreadable board %s: %s" % (p, exc), file=sys.stderr)
    why = {}
    for p in args.not_run or []:
        try:
            why.update({k: v for k, v in load_json(p).items() if isinstance(v, str)})
        except (OSError, ValueError):
            pass
    res = compare(doc, boards, why)
    v = doc["vendor"]
    md = args.out_md or os.path.join(OUT_DIR, "COMPARE-%s.md" % v)
    js = args.out_json or os.path.join(os.path.dirname(os.path.abspath(md)), "compare-%s.json" % v)
    summary = render_summary(res)
    for path, text in ((md, render_md(res)), (js, json.dumps(res, indent=1, sort_keys=True) + "\n"),
                       (args.summary or os.path.join(os.path.dirname(os.path.abspath(md)), "SUMMARY-%s.txt" % v),
                        summary + "\n")):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            fh.write(text)
    print(summary)
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="paper_compare", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("recipes", help="extract the paper's recipe set of one vendor")
    r.add_argument("--vendor", required=True, choices=VENDORS)
    r.add_argument("--board", default=None, help="the paper board (default the paper-tables board of the vendor)")
    r.add_argument("--version", default=PAPER_VERSION)
    r.add_argument("--out", default=None)
    i = sub.add_parser("ids", help="(box) the recipe races the current board plans")
    i.add_argument("--recipes", required=True)
    i.add_argument("--planner", default=os.path.join(HERE, "bench_board.py"))
    i.add_argument("--out-dir", required=True)
    s = sub.add_parser("pass-ids", help="(box) the planned races whose smoke passed")
    s.add_argument("--full-ids", required=True)
    s.add_argument("--smoke-json", action="append", default=[])
    s.add_argument("--out", required=True)
    c = sub.add_parser("compare", help="the paper's numbers against a new board")
    c.add_argument("--recipes", required=True)
    c.add_argument("--board", action="append", default=[], required=True)
    c.add_argument("--not-run", action="append", default=[],
                   help="json {race id: why} (not-planned.json, smoke-failed.json) for NOT-RUN rows")
    c.add_argument("--out-md", default=None)
    c.add_argument("--out-json", default=None)
    c.add_argument("--summary", default=None)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return {"recipes": cmd_recipes, "ids": cmd_ids, "pass-ids": cmd_pass_ids, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
