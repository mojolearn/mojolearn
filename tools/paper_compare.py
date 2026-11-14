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
  ids-all  --vendor V --planner tools/bench_board.py --out-dir DIR
        ON THE BOX, the full-board screen: every race the board plans on V (every family and lane, identical, our
        arms), in the same files as ids plus races.json (family/lane/dataset per race, for measure_owed data-keys).
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
  QUALITY-MISSING the 0.8.55 record holds a judged quality metric the new cell lacks (quality_missing lists them,
                 quality_unavailable the cell's own reason); the row cannot be judged on it. Shown, never a gate
  NOW-WINS       slower than the paper's fastest stored opponent in 0.8.55, at least as fast now
  NOW-LOSES      the reverse
  PARAMS-DIFF    the explicit parameters our arm read back differ from the paper's
  HW-DIFF        the new board's GPU is not the paper's GPU (timings are then not comparable)
With --prior-compare (the 0.8.66 compare JSON on main; a third, middle column):
  REGRESSION-VS-PRIOR / FASTER-VS-PRIOR   new / prior median ms above 1.10 / below 0.90
  RECOVERED      the prior was a REGRESSION vs 0.8.55 and the new board is SAME or FASTER vs 0.8.55
  STILL-REGRESSED the prior was a REGRESSION vs 0.8.55 and the new board still is
With --best-ever [FILE] (default bench/results/board_best_ever/<vendor>.json, tools/board_best_ever.py mine): a
"best ever (NV)" / "(AMD)" column, our best time ever on that race key (the same device when it has one), with its
date and release, and new / best-ever:
  BEST-EVER-REGRESSION  new / best ever > 1.10 with the same explicit parameters and the same output hash
  BEST-EVER-REGRESSION-UNCHECKED  new / best ever > 1.10, nothing differs, but the params or the hash of one side
                 were not recorded (boards before the parameter read-back)
  PARAMS-DIFFER  the best-ever cell's parameters (a shared key) or output hash differ from the new cell's: a slower
                 time may be a parameter or algorithm change, not a regression
  BEST-EVER-SUSPECT  new / best ever > 1.10 but the best ever is below 5% of the row's median time (board_best_ever.suspect_reason)
  BEST-EVER-HW-DIFF  the best ever was measured on another device than the new board's
Every our-arm cell of the new boards outside the recipes (a full-board screen, tools/ops/paper_compare_job.sh
board-all) gets an extra row in its own table with the same best-ever column; the recipe rows and their flags are
unchanged.
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
import board_best_ever as BE  # noqa: E402

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
    path = str(path)  # callers pass str or Path (measure_candidate stage crashed on a PosixPath, 2026-11-13)
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


#: a cell's explicit reason for a quality metric it could not record (bench_board_neural, bench_board_infer)
UNAVAILABLE = "quality_unavailable"


def shown_quality(quality):
    """QUALITY-MISSING FIX (2026-11-13, candidate ba08dac006 L40S): what the compare SHOWS of a cell's quality: the
    judged metrics, every other finite number the scorer recorded (the own-host reference distances of the forward
    and decode lanes, program-run's max_abs_err_vs_fp64) and the cell's quality_unavailable reason. Before, only the
    judged metrics were kept, so those rows read new_quality={} although the board held their quality."""
    q = quality or {}
    out = {k: v for k, v in q.items() if number(v) and not isinstance(v, bool)}
    out.update(scored_quality(q))
    if q.get(UNAVAILABLE):
        out[UNAVAILABLE] = str(q[UNAVAILABLE])[:600]
    return out


def quality_missing(old_quality, new_quality):
    """The judged metrics the old record holds and the new cell does not (QUALITY-MISSING; shown, never a gate)."""
    return sorted(set(scored_quality(old_quality)) - set(scored_quality(new_quality)))


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


def plan_all(planner, vendor):
    """The full-board screen (tools/ops/paper_compare_job.sh board-all): (full ids, smoke ids, {id: why not},
    board args, races) of EVERY race the board plans on the vendor (every family and lane, the board's datasets,
    identical, our arms; no --gpu-opponents-only: every own configuration is screened)."""
    modes = planner.modes_for(vendor, "identical")
    families = tuple(planner.FAMILIES)

    def plan(rows, shape):
        return {r["id"]: r for r in planner.plan_races(vendor, modes, families, None, planner.DATASETS, rows, shape)}

    full = plan(None, "full")
    smoke_rows = int(getattr(planner, "SMOKE_ROWS", 2000))
    smoke = plan(smoke_rows, SMOKE_SHAPE)
    full_ids, smoke_ids, missing = [], [], {}
    for rid in sorted(full):
        sid = smoke_id(rid, smoke_rows)
        if sid not in smoke:
            missing[rid] = "the board plans no smoke race %s" % sid
            continue
        full_ids.append(rid)
        smoke_ids.append(sid)
    bargs = ["--vendor", vendor, "--families", ",".join(families), "--modes", "identical",
             "--datasets", ",".join(planner.DATASETS)]
    races = [{"race_id": rid, "family": full[rid].get("family"), "lane": full[rid].get("lane"),
              "dataset": full[rid].get("dataset")} for rid in full_ids]
    return full_ids, smoke_ids, missing, bargs, races


def cmd_ids_all(args):
    planner = load_planner(args.planner)
    full, smoke, missing, bargs, races = plan_all(planner, args.vendor)
    os.makedirs(args.out_dir, exist_ok=True)
    for name, lines in (("race-ids-full.txt", full), ("race-ids-smoke.txt", smoke), ("board-args.txt", bargs)):
        with open(os.path.join(args.out_dir, name), "w") as fh:
            fh.write("".join(x + "\n" for x in lines))
    with open(os.path.join(args.out_dir, "not-planned.json"), "w") as fh:
        json.dump(missing, fh, indent=1, sort_keys=True)
    # the shape tools/measure_owed.py data-keys reads as --paper-recipes (family, lane, dataset per race)
    with open(os.path.join(args.out_dir, "races.json"), "w") as fh:
        json.dump({"recipes": races}, fh, indent=1, sort_keys=True)
    print("PAPER-IDS-ALL vendor=%s planned=%d no_smoke=%d" % (args.vendor, len(full), len(missing)))
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
           "old_quality": shown_quality(old["quality"]), "flags": []}
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
               new_quality=shown_quality(cell.get("quality")))
    missing = quality_missing(old["quality"], cell.get("quality"))
    if missing:
        # QUALITY-MISSING (2026-11-13): the 0.8.55 record holds a judged metric the new cell lacks; the row cannot be
        # judged on it. Shown with the cell's own reason when it gave one; never a gate.
        row["flags"].append("QUALITY-MISSING")
        row["quality_missing"] = missing
        row["quality_unavailable"] = (str((cell.get("quality") or {}).get(UNAVAILABLE) or "")[:600]
                                      or "the new cell recorded no reason (its scorer did not record it)")
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


VENDOR_TAG = {"nvidia": "NV", "amd": "AMD"}


def new_cell_records(boards, vendor):
    """{race key: our identical cell record} of the new boards (tools/board_best_ever.cell_records; later boards
    win, as in merge_boards)."""
    out = {}
    for b in boards:
        for rec in BE.cell_records(b):
            if rec["vendor"] == vendor:
                out[rec["key"]] = rec
    return out


def attach_best_ever(row, new_rec, best_rows):
    """The best-ever column of one row (module docstring). NEW_REC: the new cell's record or None."""
    b = (best_rows or {}).get(row["key"])
    row.update(best_ever_ms=None, best_ever_date=None, best_ever_release=None, best_ever_commit=None,
               best_ever_device=None, best_ever_hash=None, ratio_vs_best_ever=None, best_ever_params_differ=[],
               best_ever_hash_differs=False)
    if not b:
        return row
    device = (new_rec or {}).get("device")
    best = (b.get("best_by_device") or {}).get(device) if device else None
    best = best or b["best"]
    timeline = b.get("timeline") or []
    suspect = (b["best"].get("suspect") if best is b["best"] else None) \
        or BE.suspect_reason(best, [t for t in timeline if t.get("ms") != best.get("ms")] + [best])
    row["best_ever_suspect"] = suspect
    row.update(best_ever_ms=best["ms"], best_ever_date=best.get("date"), best_ever_release=best.get("release"),
               best_ever_commit=best.get("commit"), best_ever_device=best.get("device"),
               best_ever_hash=best.get("hash"))
    if not new_rec or not number(row.get("new_ms")):
        return row
    pd = BE.params_differ(best.get("params"), new_rec.get("params"))
    hd = bool(best.get("hash") and new_rec.get("hash") and best["hash"] != new_rec["hash"])
    row["best_ever_params_differ"], row["best_ever_hash_differs"] = pd, hd
    ratio = row["new_ms"] / best["ms"] if best["ms"] else None
    row["ratio_vs_best_ever"] = ratio
    if pd or hd:
        row["flags"].append("PARAMS-DIFFER")
    elif ratio is not None and ratio > SLOWER:
        known = (isinstance(best.get("params"), dict) and isinstance(new_rec.get("params"), dict)
                 and best.get("hash") and new_rec.get("hash"))
        if suspect:
            row["flags"].append("BEST-EVER-SUSPECT")
        else:
            row["flags"].append("BEST-EVER-REGRESSION" if known else "BEST-EVER-REGRESSION-UNCHECKED")
    if device and best.get("device") and best["device"] != device:
        row["flags"].append("BEST-EVER-HW-DIFF")
    return row


def extra_row(rec):
    """A row for a new-board cell outside the recipes (the full-board screen): new time and best ever only."""
    return {"key": rec["key"], "race_id": rec["race_id"], "family": rec.get("family"), "lane": rec.get("lane"),
            "dataset": rec.get("dataset"), "phase": rec["phase"], "arm": rec["arm"], "batch": rec.get("batch"),
            "new_ms": rec["ms"], "new_hash": rec.get("hash"), "new_device": rec.get("device"), "flags": []}


def compare(doc, boards, not_run_why=None, prior=None, prior_label=None, best_ever=None):
    """PRIOR: {recipe key: row} of the previous release's compare JSON (the middle column), or None.
    BEST_EVER: a tools/board_best_ever.py document of the compare's vendor (the best-ever column), or None."""
    races, gpus, versions = merge_boards(boards)
    candidate = board_candidate(boards)
    paper_gpu = doc["source"].get("gpu")
    hw_diff = bool(gpus) and paper_gpu is not None and any(g != paper_gpu for g in gpus)
    rows = [compare_one(rec, races, not_run_why or {}, hw_diff) for rec in doc["recipes"]]
    if prior is not None:
        rows = [attach_prior(r, prior) for r in rows]
    extra, extra_counts, be_label = [], {}, None
    if best_ever is not None:
        if best_ever.get("vendor") != doc["vendor"]:
            raise SystemExit("paper_compare: the best-ever file is %s, the recipes %s"
                             % (best_ever.get("vendor"), doc["vendor"]))
        be_label = "best ever (%s)" % VENDOR_TAG.get(doc["vendor"], doc["vendor"])
        new_recs = new_cell_records(boards, doc["vendor"])
        best_rows = best_ever.get("rows") or {}
        rows = [attach_best_ever(r, new_recs.get(r["key"]), best_rows) for r in rows]
        recipe_keys = {r["key"] for r in doc["recipes"]}
        extra = [attach_best_ever(extra_row(rec), rec, best_rows)
                 for k, rec in sorted(new_recs.items()) if k not in recipe_keys]
        for r in extra:
            for f in r["flags"]:
                extra_counts[f] = extra_counts.get(f, 0) + 1
    counts = {}
    for r in rows:
        for f in r["flags"]:
            counts[f] = counts.get(f, 0) + 1
    return {"schema": COMPARE_SCHEMA, "vendor": doc["vendor"], "paper_version": doc["paper_version"],
            "new_versions": versions, "new_label": new_label(candidate, versions), "candidate": candidate,
            "prior_label": prior_label if prior is not None else None,
            "paper_gpu": paper_gpu, "new_gpus": gpus, "hw_diff": hw_diff,
            "thresholds": doc.get("thresholds"), "recipes": len(rows), "flag_counts": counts, "rows": rows,
            "best_ever_label": be_label, "best_ever_source": (best_ever or {}).get("mined_at_head"),
            "extra_rows": extra, "extra_flag_counts": extra_counts}


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
    judged = scored_quality(q)
    keys = sorted(judged) + sorted(k for k in q if k not in judged and number(q[k]) and not isinstance(q[k], bool))
    text = " ".join("%s=%.4g" % (k, q[k]) for k in keys[:3])
    if q.get(UNAVAILABLE):
        text = (text + " " if text else "") + "unavailable"
    return text or "-"


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
    bl = res.get("best_ever_label")
    bcols = "%s ms | best-ever date, release | new/best | " % bl if bl else ""
    lines.append("| recipe | %s ms | %snew ms | ratio | %sflags | quality %s | quality new | q | best opponent "
                 "(ms) | vs opp %s | vs opp new | note |" % (res["paper_version"], pcols, bcols, res["paper_version"],
                                                             res["paper_version"]))
    lines.append("|---|---:|%s---:|---:|%s---|---|---|---|---|---:|---:|---|" % ("---:|---:|" if pl else "",
                                                                             "---:|---|---:|" if bl else ""))
    for r in res["rows"]:
        pvals = ("%s | %s | " % (fmt_ms(r.get("prior_ms")), fmt_ratio(r.get("ratio_vs_prior")))) if pl else ""
        bvals = best_ever_cells(r) if bl else ""
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
        if r.get("quality_missing"):
            note.append("quality missing: %s (%s)" % (",".join(r["quality_missing"]),
                                                      str(r.get("quality_unavailable") or "")[:160]))
        if r.get("params_diff"):
            note.append("params: " + ",".join(r["params_diff"][:6]))
        if r.get("best_ever_params_differ") or r.get("best_ever_hash_differs"):
            note.append("vs best ever: " + ",".join(r.get("best_ever_params_differ", [])[:6] +
                                                    (["hash"] if r.get("best_ever_hash_differs") else [])))
        lines.append("| %s | %s | %s%s | %s | %s%s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["key"], fmt_ms(r.get("old_ms")), pvals, fmt_ms(r.get("new_ms")), fmt_ratio(r.get("ratio")), bvals,
            " ".join(r["flags"]), fmt_q(r.get("old_quality")), fmt_q(r.get("new_quality")),
            r.get("quality_verdict", "-"),
            "%s (%s)" % (b.get("arm"), fmt_ms(b.get("opponent_ms"))) if b else "-",
            fmt_ratio(b.get("ratio")), fmt_ratio(b.get("ratio_new")), "; ".join(note).replace("|", "/")))
    lines.append("")
    if bl:
        ex = res.get("extra_rows") or []
        lines.append("## Rows outside the paper's recipes: %d (new against %s)" % (len(ex), bl))
        lines.append("")
        if res.get("extra_flag_counts"):
            lines.append("Flags: " + ", ".join("%s %d" % kv for kv in sorted(res["extra_flag_counts"].items())))
            lines.append("")
        lines.append("| race | new ms | %s ms | best-ever date, release | new/best | flags | note |" % bl)
        lines.append("|---|---:|---:|---|---:|---|---|")
        for r in ex:
            note = ""
            if r.get("best_ever_params_differ") or r.get("best_ever_hash_differs"):
                note = "vs best ever: " + ",".join(r.get("best_ever_params_differ", [])[:6] +
                                                   (["hash"] if r.get("best_ever_hash_differs") else []))
            elif r.get("best_ever_ms") is None:
                note = "no earlier measurement"
            lines.append("| %s | %s | %s%s | %s |" % (r["key"], fmt_ms(r.get("new_ms")), best_ever_cells(r),
                                                       " ".join(r["flags"]), note.replace("|", "/")))
        lines.append("")
    return "\n".join(lines)


def best_ever_cells(r):
    """'best ms | date, release | new/best | ' of one row."""
    when = ", ".join(str(x) for x in (r.get("best_ever_date"), r.get("best_ever_release")) if x) or "-"
    return "%s | %s | %s | " % (fmt_ms(r.get("best_ever_ms")), when.replace("|", "/"),
                                fmt_ratio(r.get("ratio_vs_best_ever")))


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
    if res.get("best_ever_label"):
        ex = res.get("extra_rows") or []
        out.append("%s: extra rows outside the recipes=%d; extra flags: %s" % (
            res["best_ever_label"], len(ex),
            " ".join("%s=%d" % kv for kv in sorted((res.get("extra_flag_counts") or {}).items())) or "-"))
        for label, pool in (("paper rows", rows), ("extra rows", ex)):
            hit = sorted([r for r in pool if "BEST-EVER-REGRESSION" in r["flags"]],
                         key=lambda r: -r["ratio_vs_best_ever"])
            if hit:
                out.append("worse than best ever, same params and hash (%s, %d):" % (label, len(hit)))
                out += ["  %.2fx %s (best %s ms %s %s -> %s ms)" % (
                    r["ratio_vs_best_ever"], r["key"], fmt_ms(r["best_ever_ms"]), r.get("best_ever_release"),
                    r.get("best_ever_date"), fmt_ms(r["new_ms"])) for r in hit[:top]]
    for flag in ("RECOVERED", "STILL-REGRESSED", "FIXED", "BROKEN", "QUALITY-DROP", "QUALITY-CHANGE",
                 "QUALITY-MISSING", "NOW-WINS",
                 "NOW-LOSES", "PARAMS-DIFF", "NOT-RUN", "BEST-EVER-REGRESSION-UNCHECKED", "PARAMS-DIFFER"):
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
    best = None
    if args.best_ever:
        best = BE.load(os.path.join(REPO, "bench", "results", "board_best_ever", "%s.json" % doc["vendor"])
                       if args.best_ever == "auto" else args.best_ever)
    res = compare(doc, boards, why, prior, plabel, best)
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
    a = sub.add_parser("ids-all", help="(box) every race the board plans: the full-board screen")
    a.add_argument("--vendor", required=True, choices=VENDORS)
    a.add_argument("--planner", default=os.path.join(HERE, "bench_board.py"))
    a.add_argument("--out-dir", required=True)
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
    c.add_argument("--best-ever", nargs="?", const="auto", default=None, metavar="FILE",
                   help="the best-ever column (tools/board_best_ever.py mine; no FILE: "
                        "bench/results/board_best_ever/<vendor>.json): BEST-EVER-REGRESSION, PARAMS-DIFFER, and a "
                        "row for every new cell outside the recipes")
    c.add_argument("--out-md", default=None)
    c.add_argument("--out-json", default=None)
    c.add_argument("--summary", default=None)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return {"recipes": cmd_recipes, "ids": cmd_ids, "ids-all": cmd_ids_all, "pass-ids": cmd_pass_ids,
            "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
