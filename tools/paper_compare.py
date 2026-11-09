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
With --prior-compare (the 0.8.66 compare JSON on main; a third, middle column):
  REGRESSION-VS-PRIOR / FASTER-VS-PRIOR   new / prior median ms above 1.10 / below 0.90
  RECOVERED      the prior was a REGRESSION vs 0.8.55 and the new board is SAME or FASTER vs 0.8.55
  STILL-REGRESSED the prior was a REGRESSION vs 0.8.55 and the new board still is
A board made from candidate wheels (bench_board --candidate-wheel; box.mojolearn.wheel.source = candidate) is
labelled "candidate <sha12> (<version>)" everywhere, with its source commit and wheel sha256s, never as a release.

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


def race_smoke_key(rid):
    """bench_board.smoke_key of a race id: family/lane/dataset, the key smoke.json files its races under and the
    board's smoke gate reads (the same race at smoke and at full size)."""
    return "/".join(rid.split("/")[:3])


def passed_full_ids(full_ids, smoke_docs):
    """(full ids whose smoke passed, {full id: why not}). Matched on the smoke key (family/lane/dataset), as the
    board's own smoke gate does: the 0.8.66 NVIDIA run lost 15 of 62 races to "no smoke result" when this
    matched the smoke id turned back into a full id (taxi classical lanes, int8 gemm, decode lanes, ...)."""
    passed, seen = set(), {}
    for doc in smoke_docs:
        for key, v in (doc.get("races") or {}).items():
            k = race_smoke_key(v.get("id") or key) if "/" in (v.get("id") or key) else key
            seen[k] = v
            if v.get("pass") is True:
                passed.add(k)
    keep, failed = [], {}
    for rid in full_ids:
        k = race_smoke_key(rid)
        if k in passed:
            keep.append(rid)
        elif k in seen:
            v = seen[k]
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


def board_candidate(boards):
    """The candidate identity of the new boards (bench_board --candidate-wheel: box.mojolearn.wheel carries the
    candidate manifest), or None for a published release. Two different candidates in one compare refuse."""
    found = {}
    for b in boards:
        wheel = (((b.get("box") or {}).get("mojolearn") or {}).get("wheel") or {})
        man = wheel.get("candidate") if wheel.get("source") == "candidate" else None
        if not man:
            continue
        ident = {"source_commit": man.get("source_commit"), "version": man.get("version"),
                 "vendor": man.get("vendor"),
                 "wheels": {f.get("distribution"): {"file": f.get("file"), "sha256": f.get("sha256")}
                            for f in man.get("files") or []}}
        found[json.dumps(ident, sort_keys=True)] = ident
    if len(found) > 1:
        raise SystemExit("paper_compare: the boards hold %d different candidates; compare one at a time"
                         % len(found))
    return next(iter(found.values()), None)


def new_label(candidate, versions):
    """'candidate <sha12> (<version>)' for a candidate build, so it is never read as a release."""
    if candidate:
        return "candidate %s (%s)" % ((candidate.get("source_commit") or "?")[:12], candidate.get("version"))
    return ",".join(versions) or "?"


def attach_prior(row, prior):
    """The middle column (the previous release, e.g. 0.8.66, from its compare JSON on main): its ms and flags,
    the new / prior ratio and the flags vs the prior:
      REGRESSION-VS-PRIOR new / prior > 1.10      FASTER-VS-PRIOR new / prior < 0.90
      RECOVERED           the prior was a REGRESSION vs 0.8.55; new is SAME or FASTER vs 0.8.55
      STILL-REGRESSED     the prior was a REGRESSION vs 0.8.55 and new still is"""
    p = prior.get(row["key"]) if prior else None
    row["prior_ms"] = p.get("new_ms") if p and number(p.get("new_ms")) else None
    row["prior_flags"] = list(p.get("flags") or []) if p else []
    row["prior_status"] = p.get("new_status") if p else "not in the prior compare"
    r = (row["new_ms"] / row["prior_ms"]) if number(row.get("new_ms")) and row["prior_ms"] else None
    row["ratio_vs_prior"] = r
    if r is not None:
        if r > SLOWER:
            row["flags"].append("REGRESSION-VS-PRIOR")
        elif r < FASTER:
            row["flags"].append("FASTER-VS-PRIOR")
    if "REGRESSION" in row["prior_flags"]:
        if "SAME" in row["flags"] or "FASTER" in row["flags"]:
            row["flags"].append("RECOVERED")
        elif "REGRESSION" in row["flags"]:
            row["flags"].append("STILL-REGRESSED")
    return row


def compare(doc, boards, not_run_why=None, prior=None, prior_label=None):
    """PRIOR: {recipe key: row} of the previous release's compare JSON (the middle column), or None."""
    races, gpus, versions = merge_boards(boards)
    candidate = board_candidate(boards)
    paper_gpu = doc["source"].get("gpu")
    hw_diff = bool(gpus) and paper_gpu is not None and any(g != paper_gpu for g in gpus)
    rows = [compare_one(rec, races, not_run_why or {}, hw_diff) for rec in doc["recipes"]]
    if prior is not None:
        rows = [attach_prior(r, prior) for r in rows]
    counts = {}
    for r in rows:
        for f in r["flags"]:
            counts[f] = counts.get(f, 0) + 1
    return {"schema": COMPARE_SCHEMA, "vendor": doc["vendor"], "paper_version": doc["paper_version"],
            "new_versions": versions, "new_label": new_label(candidate, versions), "candidate": candidate,
            "prior_label": prior_label if prior is not None else None,
            "paper_gpu": paper_gpu, "new_gpus": gpus, "hw_diff": hw_diff,
            "thresholds": doc.get("thresholds"), "recipes": len(rows), "flag_counts": counts, "rows": rows}


def load_prior(path):
    """(rows by key, label) of a previous compare JSON (bench/results/paper_compare/compare-<vendor>.json)."""
    doc = load_json(path)
    if doc.get("schema") != COMPARE_SCHEMA:
        raise SystemExit("paper_compare: %s is not a %s document" % (path, COMPARE_SCHEMA))
    if doc.get("candidate"):
        raise SystemExit("paper_compare: the prior column must be a published release, not %s"
                         % doc.get("new_label"))
    return {r["key"]: r for r in doc.get("rows") or []}, ",".join(doc.get("new_versions") or []) or "prior"


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
    v, new = res["vendor"], res.get("new_label") or ", ".join(res["new_versions"]) or "?"
    pl = res.get("prior_label")
    lines = ["# Paper measurements: %s%s against %s on %s" % (res["paper_version"], ", " + pl if pl else "", new, v),
             ""]
    cand = res.get("candidate")
    if cand:
        lines.append("**Candidate build, not a release**: source %s, version %s; wheels %s." % (
            cand.get("source_commit"), cand.get("version"),
            ", ".join("%s sha256 %s" % (w.get("file"), w.get("sha256")) for _, w in sorted(cand["wheels"].items()))))
        lines.append("")
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
    pcols = "%s ms | new/%s | " % (pl, pl) if pl else ""
    lines.append("| recipe | %s ms | %snew ms | ratio | flags | quality %s | quality new | q | best opponent "
                 "(ms) | vs opp %s | vs opp new | note |" % (res["paper_version"], pcols, res["paper_version"],
                                                             res["paper_version"]))
    lines.append("|---|---:|%s---:|---:|---|---|---|---|---|---:|---:|---|" % ("---:|---:|" if pl else ""))
    for r in res["rows"]:
        pvals = ("%s | %s | " % (fmt_ms(r.get("prior_ms")), fmt_ratio(r.get("ratio_vs_prior")))) if pl else ""
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
        lines.append("| %s | %s | %s%s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["key"], fmt_ms(r.get("old_ms")), pvals, fmt_ms(r.get("new_ms")), fmt_ratio(r.get("ratio")),
            " ".join(r["flags"]), fmt_q(r.get("old_quality")), fmt_q(r.get("new_quality")),
            r.get("quality_verdict", "-"),
            "%s (%s)" % (b.get("arm"), fmt_ms(b.get("opponent_ms"))) if b else "-",
            fmt_ratio(b.get("ratio")), fmt_ratio(b.get("ratio_new")), "; ".join(note).replace("|", "/")))
    lines.append("")
    return "\n".join(lines)


def render_summary(res, top=8):
    rows = res["rows"]
    label = res.get("new_label") or ",".join(res["new_versions"]) or "?"
    out = ["PAPER-COMPARE vendor=%s %s->%s%s recipes=%d gpu %s -> %s%s" % (
        res["vendor"], res["paper_version"], (res["prior_label"] + "->") if res.get("prior_label") else "",
        label.replace(" ", "_"), res["recipes"],
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
    if res.get("prior_label"):
        worse = sorted([r for r in rows if "REGRESSION-VS-PRIOR" in r["flags"]], key=lambda r: -r["ratio_vs_prior"])
        if worse:
            out.append("worse than %s:" % res["prior_label"])
            out += ["  %.2fx %s (%s -> %s ms)" % (r["ratio_vs_prior"], r["key"], fmt_ms(r["prior_ms"]),
                                                  fmt_ms(r["new_ms"])) for r in worse[:top]]
    for flag in ("RECOVERED", "STILL-REGRESSED", "FIXED", "BROKEN", "QUALITY-DROP", "QUALITY-CHANGE", "NOW-WINS",
                 "NOW-LOSES", "PARAMS-DIFF", "NOT-RUN"):
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
    prior, plabel = (None, None)
    if args.prior_compare:
        prior, plabel = load_prior(args.prior_compare)
        if args.prior_label:
            plabel = args.prior_label
    res = compare(doc, boards, why, prior, plabel)
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
    c.add_argument("--prior-compare", default=None,
                   help="the previous release's compare JSON (bench/results/paper_compare/compare-<vendor>.json, "
                        "0.8.66): the middle column, REGRESSION-VS-PRIOR / RECOVERED / STILL-REGRESSED")
    c.add_argument("--prior-label", default=None, help="the middle column's name (default its new_versions)")
    c.add_argument("--out-md", default=None)
    c.add_argument("--out-json", default=None)
    c.add_argument("--summary", default=None)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return {"recipes": cmd_recipes, "ids": cmd_ids, "pass-ids": cmd_pass_ids, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
