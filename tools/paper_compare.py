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
  compare  --recipes JSON --board BOARD.json... [--prior-board VERSION=BOARD.json ...] [--reference-compare JSON]
           [--best-ever [FILE]] [--out-md MD] [--out-json JSON] [--summary TXT]
        the new board against OUR prior wheels re-timed in the same job on the same box, per recipe: ms of each,
        ratios, quality, the flags below, and the stored columns (the paper record, a stored compare, the stored
        opponents, the best ever) as reference only. Prints a one-screen summary with both arms' host probes.

SAME BOX ONLY (AGENTS.md "Times are compared only on the same box", Andrew, 2026-11-14; the 2026-11-14 compare of
80a7f8aca3 on L40S host 9204fff9f8f7 against stored numbers from host 1e08d01c3d48 raised 56 false REGRESSION flags
on code that had not changed). Every board carries the box's host probe (tools/ops/host_probe.py through
bench_board --host-probe: box.host_probe and each race's host.host_id). A time flag is derived only from two cells
whose boards carry the same host id. A --prior-board from another host REFUSES the compare (exit non-zero); a board
without a probe gives no verdict (NOT-COMPARED). The stored paper record (old_ms in the recipes: ref_ms here), a
stored compare JSON (--reference-compare: ref_prior_ms), the paper's opponent cells and the best-ever file are
shown with their host and never make a flag.

FLAGS (per recipe; a recipe can carry several)
  REGRESSION     new / baseline median_ms > 1.10 (slower)     FASTER   new / baseline < 0.90
  SAME           0.90 <= ratio <= 1.10
                 (baseline = the --prior-board of the paper's version, 0.8.55, re-timed on this box)
  FIXED          the baseline cell has no time (refused, failed) and the new board has one
  BROKEN         the baseline has a time and the new board has none (its status is shown)
  STILL-FAILING  neither has a time
  BASELINE-NOT-RUN  the baseline board has no cell for the race (its smoke failed, it does not plan it, the job did
                 not reach it): NOT-RUN for that column, never a regression
  NOT-COMPARED   no on-box baseline (no --prior-board of the paper's version, or no host probe on either side)
  NOT-RUN        no new cell: the board no longer plans the race, its smoke failed, or the job did not reach it
  QUALITY-DROP   tools/board_quality.compare(new, paper record) FAILs on a one-sided score or a reference bound:
                 worse than 0.8.55 beyond the board's declared margin (the quality-parity policy)
  QUALITY-CHANGE the same policy FAILs on a symmetric metric (no better side)
  QUALITY-MISSING the 0.8.55 record holds a judged quality metric the new cell lacks (quality_missing lists them,
                 quality_unavailable the cell's own reason); the row cannot be judged on it. Shown, never a gate
  NOW-WINS / NOW-LOSES  vs the paper's fastest stored opponent: only when the stored opponents ran on this box's
                 host (never, for the paper's boards; the ratio is shown)
  PARAMS-DIFF    the explicit parameters our arm read back differ from the paper's
  HW-DIFF        the new board's GPU is not the paper's GPU
With more --prior-board versions (default job: 0.8.55 and 0.8.66), each gets a column (ms, new / prior); the last
one is the prior release:
  REGRESSION-VS-PRIOR / FASTER-VS-PRIOR   new / prior median ms above 1.10 / below 0.90
  RECOVERED      the prior was > 1.10 x the on-box baseline and the new board is SAME or FASTER vs that baseline
  STILL-REGRESSED the prior was > 1.10 x the on-box baseline and the new board still is
With --best-ever [FILE] (default bench/results/board_best_ever/<vendor>.json, tools/board_best_ever.py mine): a
"best ever (NV)" / "(AMD)" reference column, our best time ever on that race key, with its date, release and host,
and new / best-ever. Its time flags need the same host id on the best-ever cell (records mined before 2026-11-14
carry none, so they are shown and never flagged):
  BEST-EVER-REGRESSION  new / best ever > 1.10 with the same explicit parameters and the same output hash
  BEST-EVER-REGRESSION-UNCHECKED  the same, but the params or the hash of one side were not recorded
  BEST-EVER-SUSPECT  new / best ever > 1.10 but the best ever is below 5% of the row's median time
  PARAMS-DIFFER  the best-ever cell's parameters (a shared key) or output hash differ from the new cell's (not a time)
  BEST-EVER-HW-DIFF  the best ever was measured on another device than the new board's
Every our-arm cell of the new boards outside the recipes (a full-board screen, tools/ops/paper_compare_job.sh
board-all) gets an extra row in its own table with the on-box prior columns (REGRESSION / FASTER vs the baseline,
REGRESSION-VS-PRIOR) and the best-ever reference.
A board made from candidate wheels (bench_board --candidate-wheel; box.mojolearn.wheel.source = candidate) is
labelled "candidate <sha12> (<version>)" everywhere, with its source commit and wheel sha256s, never as a release.

The new numbers come from tools/ops/paper_compare_job.sh, which runs these recipes through tools/bench_board.py
(our arms only) for the candidate and for each prior wheel from PyPI (PC_PRIOR_VERSIONS) on the same box in the same
job. Tests: tools/test_paper_compare.py, tools/test_paper_compare_same_box.py.
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

def write_atomic(path, text):
    """Write a results file whole or not at all: a temp file beside it, fsync, then rename over the target.

    ATOMIC-WRITE FIX (2026-11-14, L40S pod nvc1): the owed report died on a truncated JSON written while the disk
    was full (JSONDecodeError line 1 col 87). A reader now sees the old file or the complete new one; a failed write
    (ENOSPC) raises here and leaves the target untouched."""
    path = os.path.abspath(str(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = "%s.tmp-%d" % (path, os.getpid())
    try:
        with open(tmp, "w") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def json_text(obj):
    return json.dumps(obj, indent=1, sort_keys=True) + "\n"


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
# the box a time was taken on (AGENTS.md "Times are compared only on the same box", Andrew, 2026-11-14)
# ---------------------------------------------------------------------------------------------------------------

def board_probe(board):
    """The tools/ops/host_probe.py document a board recorded (bench_board --host-probe), or None."""
    p = (board.get("box") or {}).get("host_probe")
    return p if isinstance(p, dict) else None


def board_host_ids(board):
    """Every host id a board carries: its box probe's and each race's host.host_id (bench_board race_host)."""
    ids = set()
    p = board_probe(board)
    if p and p.get("host_id"):
        ids.add(p["host_id"])
    for race in (board.get("races") or {}).values():
        h = (race.get("host") or {}).get("host_id") if isinstance(race, dict) else None
        if h:
            ids.add(h)
    return ids


def host_label(board):
    """The host a stored board ran on, for a reference column: its probe's host id, else its hostname."""
    p = board_probe(board)
    if p and p.get("host_id"):
        return p["host_id"]
    return ((board.get("box") or {}).get("host") or {}).get("hostname")


def candidate_host(cand_boards):
    """(host id, None) of the candidate boards, or (None, why) without a probe. Two hosts refuse (SystemExit)."""
    cand = set().union(*[board_host_ids(b) for b in cand_boards]) if cand_boards else set()
    if len(cand) > 1:
        raise SystemExit("paper_compare: REFUSED: the candidate boards ran on %d hosts (%s); times are compared only "
                         "on the same box" % (len(cand), ", ".join(sorted(cand))))
    return (next(iter(cand)), None) if cand else (None, "the candidate board carries no host probe")


def same_box(cand_boards, other_boards, what):
    """(host id, None) when the candidate boards and OTHER_BOARDS carry one host id; (None, why) when either side
    has none (no probe: the column is shown and makes no flag). Two host ids refuse (SystemExit): a compare across
    hosts is never a verdict (2026-11-14: 56 false REGRESSION flags from two L40S hosts)."""
    host, why = candidate_host(cand_boards)
    other = set().union(*[board_host_ids(b) for b in other_boards]) if other_boards else set()
    if host and other and other != {host}:
        raise SystemExit("paper_compare: REFUSED: %s ran on host %s, the candidate on %s; times are compared only on "
                         "the same box (AGENTS.md)" % (what, ", ".join(sorted(other)), host))
    if not host:
        return None, why
    if not other:
        return None, "%s carries no host probe" % what
    return host, None


def probe_line(probe):
    """One line of a host probe for the summary (both arms' probes are printed beside the verdict)."""
    if not probe:
        return "no host probe"
    pcie = probe.get("pcie") or {}
    link = ("gen%s x%s%s" % (pcie.get("gen"), pcie.get("width"), "" if pcie.get("read_during_copy") else " (idle)")
            if pcie.get("gen") is not None else "pcie ?")
    g = lambda k: "-" if probe.get(k) is None else probe.get(k)  # noqa: E731
    return ("host %s; cpu %s; numa %s; cores %s (cgroup %s); memcpy %s GB/s; h2d pageable %s / pinned %s GB/s; %s; "
            "gpu %s driver %s" % (g("host_id"), g("cpu_model"), g("numa_nodes"), g("affinity_cores") if probe.get(
                "affinity_cores") is not None else g("nproc"), g("cgroup_cores"), g("memcpy_gbps"),
                g("h2d_pageable_gbps"), g("h2d_pinned_gbps"), link, (probe.get("gpu") or {}).get("name"),
                (probe.get("gpu") or {}).get("driver")))


def parse_prior_board(text):
    """'<version>=<board.json>' (compare --prior-board) -> (version, path)."""
    v, sep, path = text.partition("=")
    if not sep or not v or not path:
        raise SystemExit("paper_compare: --prior-board wants <version>=<board.json>, not %r" % text)
    return v, path


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
            # the stored column's host (reference only: never a flag against a time from another host)
            "host": host_label(board),
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
    write_atomic(out, json_text(doc))
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
    write_atomic(os.path.join(args.out_dir, "race-ids-full.txt"), "".join(x + "\n" for x in full))
    write_atomic(os.path.join(args.out_dir, "race-ids-smoke.txt"), "".join(x + "\n" for x in smoke))
    write_atomic(os.path.join(args.out_dir, "board-args.txt"),
                 "".join(x + "\n" for x in board_args(doc["bench_board_args"])))
    write_atomic(os.path.join(args.out_dir, "not-planned.json"), json_text(missing))
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
        write_atomic(os.path.join(args.out_dir, name), "".join(x + "\n" for x in lines))
    write_atomic(os.path.join(args.out_dir, "not-planned.json"), json_text(missing))
    # the shape tools/measure_owed.py data-keys reads as --paper-recipes (family, lane, dataset per race)
    write_atomic(os.path.join(args.out_dir, "races.json"), json_text({"recipes": races}))
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
    write_atomic(args.out, "".join(x + "\n" for x in keep))
    write_atomic(os.path.join(os.path.dirname(os.path.abspath(args.out)), "smoke-failed.json"), json_text(failed))
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


#: lane fix-gbdt-prebinned-infer (2026-11-13; PARAMS-DIFF on trees/gbdt-depthwise/istella ours-prebinned, every
#: cell): the pre-binned `input_binning` record gained "; <n> training rows" on 2026-10-30 (FAIRNESS FIX,
#: tools/speed_gbdt_arm.py PREBIN_ROWS, read by the LightGBM lossguide mapping) after the 0.8.55 paper record was
#: taken. The borders, the border count and the bins sha256 (which hashes both binned blocks, so it fixes the row
#: count too) are unchanged, so the clause is spelling only for this comparison; any other change still differs.
_PREBIN_ROWS_CLAUSE = re.compile(r"; \d+ training rows(?=;)")


def _param_value(key, value):
    if key == "input_binning" and isinstance(value, str):
        return _PREBIN_ROWS_CLAUSE.sub("", value)
    return value


def params_diff(old, new):
    if old is None or new is None:
        return []
    keys = sorted(set(old) | set(new))
    return [k for k in keys if _param_value(k, old.get(k)) != _param_value(k, new.get(k))]


def same_clock_ms(opp, cell, phase):
    """Our new time on the opponent comparison's clock: whole = median_ms, kernel = the derived kernel span."""
    if cell is None:
        return None
    if opp.get("clock") == "kernel":
        return kernel_ms(cell) if phase == "fit" else None
    if opp.get("clock") == "whole":
        return cell.get("median_ms") if number(cell.get("median_ms")) else None
    return None       # MIXED clocks: the paper shows no ratio; neither do we


def compare_one(rec, races, not_run_why, hw_diff, base=None, same_host_opponents=False):
    """One recipe row. BASE: the on-box baseline (the paper's version re-timed in the same job on the same box):
    {"races": {...}, "label": v}, or None. Every time flag (REGRESSION, FASTER, SAME, FIXED, BROKEN,
    STILL-FAILING) comes from BASE only; the stored paper record is the reference column (ref_ms, with its host) and
    never makes a flag (AGENTS.md "Times are compared only on the same box", 2026-11-14). Without BASE the row is
    NOT-COMPARED. Quality and parameters are still judged against the paper record (they are not times).
    SAME_HOST_OPPONENTS: the stored opponents were timed on this box (NOW-WINS / NOW-LOSES may then be derived)."""
    old = rec["v0855"]
    race = races.get(rec["race_id"])
    cell = find_cell(race, rec["phase"], rec["arm"], rec["batch"]) if race else None
    ref_t = timed(old["status"], old["median_ms"])
    row = {"key": rec["key"], "race_id": rec["race_id"], "family": rec["family"], "lane": rec["lane"],
           "dataset": rec["dataset"], "phase": rec["phase"], "arm": rec["arm"], "batch": rec["batch"],
           "ref_ms": old["median_ms"] if ref_t else None, "ref_status": old["status"],
           "old_ms": None, "old_status": "no on-box baseline" if base is None else "not run on this box",
           "old_quality": shown_quality(old["quality"]), "flags": []}
    if base is not None:
        brace = (base.get("races") or {}).get(rec["race_id"])
        bcell = find_cell(brace, rec["phase"], rec["arm"], rec["batch"]) if brace else None
        if bcell is not None:
            row["old_status"] = short_status(bcell.get("status"))
            if timed(row["old_status"], bcell.get("median_ms")):
                row["old_ms"] = bcell.get("median_ms")
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
    missing = quality_missing(old["quality"], cell.get("quality")) if new_t else []   # untimed: BROKEN says it
    if missing:
        # QUALITY-MISSING (2026-11-13): the 0.8.55 record holds a judged metric the new cell lacks; the row cannot be
        # judged on it. Shown with the cell's own reason when it gave one; never a gate.
        row["flags"].append("QUALITY-MISSING")
        row["quality_missing"] = missing
        row["quality_unavailable"] = (str((cell.get("quality") or {}).get(UNAVAILABLE) or "")[:600]
                                      or "the new cell recorded no reason (its scorer did not record it)")
    row["ratio"] = None
    row["ratio_vs_ref"] = row["new_ms"] / row["ref_ms"] if new_t and row["ref_ms"] else None   # shown, never a flag
    old_t = row["old_ms"] is not None
    if base is None:
        row["flags"].append("NOT-COMPARED")
    elif old_t and new_t:
        row["ratio"] = row["new_ms"] / row["old_ms"] if row["old_ms"] else None
        r = row["ratio"]
        row["flags"].append("REGRESSION" if r is not None and r > SLOWER else
                            "FASTER" if r is not None and r < FASTER else "SAME")
    elif row["old_status"] == "not run on this box":
        # the prior wheel has no cell for this row (its smoke failed, it refused at plan, the job did not reach it):
        # NOT-RUN for that column, never a regression (brief same-box-compare, 2026-11-14)
        row["flags"].append("BASELINE-NOT-RUN")
    else:
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
    # opponents: the paper's stored numbers (their host is the paper board's), our new time on the same clock. The
    # ratio is shown; NOW-WINS / NOW-LOSES only when the stored opponent ran on this box (never across hosts).
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
        if number(b.get("ratio_new")) and same_host_opponents:
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


def attach_prior_board(row, version, races, flagged):
    """One on-box prior column (a prior release re-timed in the same job on the same box): row["priors"][version]
    = {ms, status, ratio new/prior}. FLAGGED (the last release, PC_PRIOR_VERSION, on a verified same box) adds:
      REGRESSION-VS-PRIOR new / prior > 1.10      FASTER-VS-PRIOR new / prior < 0.90
      RECOVERED           the prior was > 1.10 x the on-box baseline; new is SAME or FASTER vs that baseline
      STILL-REGRESSED     the prior was > 1.10 x the on-box baseline and new still is
    A prior without a cell (its smoke failed, it refused, the job did not reach it) is NOT-RUN in its column."""
    c = find_cell((races or {}).get(row["race_id"]), row["phase"], row["arm"], row["batch"]) \
        if (races or {}).get(row["race_id"]) else None
    status = short_status(c.get("status")) if c else "NOT-RUN on this box"
    ms = c.get("median_ms") if c and timed(status, c.get("median_ms")) else None
    r = (row["new_ms"] / ms) if number(row.get("new_ms")) and ms else None
    row.setdefault("priors", {})[version] = {"ms": ms, "status": status, "ratio": r}
    if not flagged:
        return row
    row["prior_ms"], row["prior_status"], row["ratio_vs_prior"] = ms, status, r
    if r is not None:
        if r > SLOWER:
            row["flags"].append("REGRESSION-VS-PRIOR")
        elif r < FASTER:
            row["flags"].append("FASTER-VS-PRIOR")
    if ms and number(row.get("old_ms")) and ms / row["old_ms"] > SLOWER:
        if "SAME" in row["flags"] or "FASTER" in row["flags"]:
            row["flags"].append("RECOVERED")
        elif "REGRESSION" in row["flags"]:
            row["flags"].append("STILL-REGRESSED")
    return row


def attach_reference(row, ref):
    """A stored compare's new column (another run, another host): shown as reference only, never a flag."""
    p = ref.get(row["key"]) if ref else None
    row["ref_prior_ms"] = p.get("new_ms") if p and number(p.get("new_ms")) else None
    row["ref_prior_status"] = p.get("new_status") if p else "not in the reference compare"
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


def attach_best_ever(row, new_rec, best_rows, host=None):
    """The best-ever column of one row (module docstring). NEW_REC: the new cell's record or None. HOST: the new
    board's host id. The best ever is a stored time of another run: its ratio is shown, and the time flags
    (BEST-EVER-REGRESSION, -UNCHECKED, -SUSPECT) are derived only when it carries the same host id (2026-11-14)."""
    b = (best_rows or {}).get(row["key"])
    row.update(best_ever_ms=None, best_ever_date=None, best_ever_release=None, best_ever_commit=None,
               best_ever_device=None, best_ever_hash=None, best_ever_host=None, ratio_vs_best_ever=None,
               best_ever_params_differ=[], best_ever_hash_differs=False)
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
               best_ever_hash=best.get("hash"), best_ever_host=best.get("host_id"))
    if not new_rec or not number(row.get("new_ms")):
        return row
    pd = BE.params_differ(best.get("params"), new_rec.get("params"))
    hd = bool(best.get("hash") and new_rec.get("hash") and best["hash"] != new_rec["hash"])
    row["best_ever_params_differ"], row["best_ever_hash_differs"] = pd, hd
    ratio = row["new_ms"] / best["ms"] if best["ms"] else None
    row["ratio_vs_best_ever"] = ratio
    if pd or hd:
        row["flags"].append("PARAMS-DIFFER")
    elif ratio is not None and ratio > SLOWER and host and best.get("host_id") == host:
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


def compare(doc, boards, not_run_why=None, prior_boards=None, reference=None, reference_label=None,
            best_ever=None):
    """PRIOR_BOARDS: [(version, board doc)] of OUR prior wheels re-timed in the same job on the same box
    (tools/ops/paper_compare_job.sh board-prior-<version>/): the one whose version is the paper's is the baseline
    (old_ms, the REGRESSION / FASTER / FIXED / BROKEN flags), the last other one the flagged prior column
    (REGRESSION-VS-PRIOR, RECOVERED, STILL-REGRESSED). A prior board from another host refuses (SystemExit).
    REFERENCE: {recipe key: row} of a stored compare JSON (another run): a reference column, never a flag.
    BEST_EVER: a tools/board_best_ever.py document of the compare's vendor (the best-ever column), or None."""
    races, gpus, versions = merge_boards(boards)
    candidate = board_candidate(boards)
    paper_gpu = doc["source"].get("gpu")
    hw_diff = bool(gpus) and paper_gpu is not None and any(g != paper_gpu for g in gpus)
    host, why_no_host = candidate_host(boards)
    priors = []
    for version, pb in prior_boards or []:
        h, why = same_box(boards, [pb], "the %s prior board" % version)    # refuses two hosts
        priors.append({"version": version, "races": pb.get("races") or {}, "host_id": h, "why_not": why,
                       "probe": board_probe(pb), "versions": merge_boards([pb])[2]})
    base = next((p for p in priors if p["version"] == doc["paper_version"]), None)
    others = [p for p in priors if p is not base]
    flagged = others[-1] if others else None
    verified = lambda p: p is not None and p["host_id"] is not None  # noqa: E731
    stored_host = doc["source"].get("host")
    same_host_opps = bool(host and stored_host and stored_host == host)
    rows = [compare_one(rec, races, not_run_why or {}, hw_diff,
                        {"races": base["races"]} if verified(base) else None, same_host_opps)
            for rec in doc["recipes"]]
    for p in others:
        rows = [attach_prior_board(r, p["version"], p["races"], p is flagged and verified(p)) for r in rows]
    if reference is not None:
        rows = [attach_reference(r, reference) for r in rows]
    extra, extra_counts, be_label = [], {}, None
    if best_ever is not None:
        if best_ever.get("vendor") != doc["vendor"]:
            raise SystemExit("paper_compare: the best-ever file is %s, the recipes %s"
                             % (best_ever.get("vendor"), doc["vendor"]))
        be_label = "best ever (%s)" % VENDOR_TAG.get(doc["vendor"], doc["vendor"])
        new_recs = new_cell_records(boards, doc["vendor"])
        best_rows = best_ever.get("rows") or {}
        rows = [attach_best_ever(r, new_recs.get(r["key"]), best_rows, host) for r in rows]
        recipe_keys = {r["key"] for r in doc["recipes"]}
        extra = [attach_best_ever(extra_row(rec), rec, best_rows, host)
                 for k, rec in sorted(new_recs.items()) if k not in recipe_keys]
        if others or base:
            extra = [attach_extra_priors(r, base, others, flagged, verified) for r in extra]
        for r in extra:
            for f in r["flags"]:
                extra_counts[f] = extra_counts.get(f, 0) + 1
    counts = {}
    for r in rows:
        for f in r["flags"]:
            counts[f] = counts.get(f, 0) + 1
    if verified(base):
        basis = "same box %s: %s re-timed in this job" % (host, base["version"])
    elif base is not None:
        basis = "no verdict: %s" % base["why_not"]
    else:
        basis = "no verdict: no on-box %s board (--prior-board %s=...)" % (doc["paper_version"], doc["paper_version"])
    return {"schema": COMPARE_SCHEMA, "vendor": doc["vendor"], "paper_version": doc["paper_version"],
            "new_versions": versions, "new_label": new_label(candidate, versions), "candidate": candidate,
            "host_id": host, "host_probe": next((board_probe(b) for b in boards if board_probe(b)), None),
            "host_missing": why_no_host, "verdict_basis": basis,
            "baseline_label": ("%s (this box)" % base["version"]) if base else None,
            "prior_label": flagged["version"] if flagged else None,
            "prior_columns": [{"version": p["version"], "host_id": p["host_id"], "probe": p["probe"],
                               "why_not": p["why_not"], "board_versions": p["versions"]} for p in priors],
            "reference_label": reference_label if reference is not None else None,
            "stored_host": stored_host,
            "paper_gpu": paper_gpu, "new_gpus": gpus, "hw_diff": hw_diff,
            "thresholds": doc.get("thresholds"), "recipes": len(rows), "flag_counts": counts, "rows": rows,
            "best_ever_label": be_label, "best_ever_source": (best_ever or {}).get("mined_at_head"),
            "extra_rows": extra, "extra_flag_counts": extra_counts}


def attach_extra_priors(row, base, others, flagged, verified):
    """The on-box prior columns of a full-board row outside the recipes: new / <version> per prior, and REGRESSION /
    FASTER vs the on-box baseline and the flagged prior when their host is verified (the same rule as a recipe)."""
    for p in ([base] if base else []) + others:
        attach_prior_board(row, p["version"], p["races"], False)
    if verified(base):
        r = row["priors"][base["version"]]["ratio"]
        if r is not None:
            row["flags"].append("REGRESSION" if r > SLOWER else "FASTER" if r < FASTER else "SAME")
    if verified(flagged):
        r = row["priors"][flagged["version"]]["ratio"]
        if r is not None and r > SLOWER:
            row["flags"].append("REGRESSION-VS-PRIOR")
        elif r is not None and r < FASTER:
            row["flags"].append("FASTER-VS-PRIOR")
    return row


def load_reference(path):
    """(rows by key, label) of a stored compare JSON (bench/results/paper_compare/compare-<vendor>.json): a reference
    column only. The label names the run and its host ("host not recorded" for a compare made before 2026-11-14)."""
    doc = load_json(path)
    if doc.get("schema") != COMPARE_SCHEMA:
        raise SystemExit("paper_compare: %s is not a %s document" % (path, COMPARE_SCHEMA))
    label = doc.get("new_label") or ",".join(doc.get("new_versions") or []) or "reference"
    return ({r["key"]: r for r in doc.get("rows") or []},
            "%s stored (host %s)" % (label, doc.get("host_id") or "not recorded"))


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


def other_prior_versions(res):
    """The on-box prior columns besides the baseline, in order."""
    base = (res.get("baseline_label") or "").split(" ")[0]
    return [p["version"] for p in res.get("prior_columns") or [] if p["version"] != base]


def render_md(res):
    v, new = res["vendor"], res.get("new_label") or ", ".join(res["new_versions"]) or "?"
    pv = other_prior_versions(res)
    base = res.get("baseline_label") or "%s (no on-box board)" % res["paper_version"]
    lines = ["# Paper measurements: %s%s against %s on %s" % (base, "".join(", %s (this box)" % x for x in pv), new, v),
             ""]
    cand = res.get("candidate")
    if cand:
        lines.append("**Candidate build, not a release**: source %s, version %s; wheels %s." % (
            cand.get("source_commit"), cand.get("version"),
            ", ".join("%s sha256 %s" % (w.get("file"), w.get("sha256")) for _, w in sorted(cand["wheels"].items()))))
        lines.append("")
    lines.append("**Verdict basis**: %s. Times are compared only on the same box (AGENTS.md); every stored column "
                 "below is reference only and makes no flag." % res.get("verdict_basis"))
    lines.append("")
    lines.append("- candidate box: %s" % probe_line(res.get("host_probe")))
    for p in res.get("prior_columns") or []:
        lines.append("- %s box: %s%s" % (p["version"], probe_line(p.get("probe")),
                                         " (%s)" % p["why_not"] if p.get("why_not") else ""))
    lines.append("- stored %s column: the paper board, host %s" % (res["paper_version"],
                                                                   res.get("stored_host") or "not recorded"))
    if res.get("reference_label"):
        lines.append("- stored reference column: %s" % res["reference_label"])
    lines.append("")
    lines.append("Paper GPU: %s. New GPU: %s.%s" % (res["paper_gpu"], ", ".join(res["new_gpus"]) or "?",
                 " **Different hardware: timings are not comparable.**" if res["hw_diff"] else ""))
    t = res.get("thresholds") or {}
    lines.append("Ratio = new / %s median ms on this box; REGRESSION above %s, FASTER below %s; quality judged by %s "
                 "against the paper record. Opponents are the paper's stored cells (never rerun, host %s); 'vs opp' is "
                 "ours over the paper's fastest paired opponent on that comparison's clock, shown, never a flag across "
                 "hosts." % (base, t.get("regression_ratio_above"), t.get("faster_ratio_below"),
                             t.get("quality_policy"), res.get("stored_host") or "not recorded"))
    lines.append("")
    lines.append("Flags: " + ", ".join("%s %d" % kv for kv in sorted(res["flag_counts"].items())))
    lines.append("")
    pcols = "".join("%s ms | new/%s | " % (x, x) for x in pv)
    bl = res.get("best_ever_label")
    bcols = "%s ms | best-ever date, release | new/best | " % bl if bl else ""
    rl = res.get("reference_label")
    rcols = "reference ms | " if rl else ""
    lines.append("| recipe | %s ms | %snew ms | ratio | %sflags | stored %s ms | %squality %s | quality new | q | "
                 "best opponent (ms) | vs opp %s | vs opp new | note |" % (base, pcols, bcols, res["paper_version"],
                                                                         rcols, res["paper_version"],
                                                                         res["paper_version"]))
    lines.append("|---|---:|%s---:|---:|%s---|---:|%s---|---|---|---|---:|---:|---|" % (
        "---:|---:|" * len(pv), "---:|---|---:|" if bl else "", "---:|" if rl else ""))
    for r in res["rows"]:
        pvals = "".join("%s | %s | " % (fmt_ms(((r.get("priors") or {}).get(x) or {}).get("ms")),
                                        fmt_ratio(((r.get("priors") or {}).get(x) or {}).get("ratio"))) for x in pv)
        bvals = best_ever_cells(r) if bl else ""
        rvals = "%s | " % fmt_ms(r.get("ref_prior_ms")) if rl else ""
        b = r.get("best_opponent") or {}
        note = []
        if r.get("why"):
            note.append(r["why"])
        if r.get("new_status") not in (None, "ok", "missing") and r.get("new_ms") is None:
            note.append("new: " + r["new_status"][:120])
        if r.get("old_ms") is None and r.get("old_status") != "ok":
            note.append("this box %s: %s" % (res["paper_version"], str(r["old_status"])[:120]))
        for x in pv:
            st = ((r.get("priors") or {}).get(x) or {}).get("status")
            if st and st != "ok" and ((r.get("priors") or {}).get(x) or {}).get("ms") is None:
                note.append("this box %s: %s" % (x, st[:120]))
        if r.get("ref_ms") is None and r.get("ref_status") != "ok":
            note.append("stored: " + str(r["ref_status"])[:120])
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
        lines.append("| %s | %s | %s%s | %s | %s%s | %s | %s%s | %s | %s | %s | %s | %s | %s |" % (
            r["key"], fmt_ms(r.get("old_ms")), pvals, fmt_ms(r.get("new_ms")), fmt_ratio(r.get("ratio")), bvals,
            " ".join(r["flags"]), fmt_ms(r.get("ref_ms")), rvals, fmt_q(r.get("old_quality")),
            fmt_q(r.get("new_quality")), r.get("quality_verdict", "-"),
            "%s (%s)" % (b.get("arm"), fmt_ms(b.get("opponent_ms"))) if b else "-",
            fmt_ratio(b.get("ratio")), fmt_ratio(b.get("ratio_new")), "; ".join(note).replace("|", "/")))
    lines.append("")
    if bl:
        ex = res.get("extra_rows") or []
        allp = [p["version"] for p in res.get("prior_columns") or []]
        lines.append("## Rows outside the paper's recipes: %d (new against %s)" % (
            len(ex), ", ".join(["%s (this box)" % x for x in allp] + [bl + " (reference)"])))
        lines.append("")
        if res.get("extra_flag_counts"):
            lines.append("Flags: " + ", ".join("%s %d" % kv for kv in sorted(res["extra_flag_counts"].items())))
            lines.append("")
        xcols = "".join("%s ms | new/%s | " % (x, x) for x in allp)
        lines.append("| race | new ms | %s%s ms | best-ever date, release | new/best | flags | note |" % (xcols, bl))
        lines.append("|---|---:|%s---:|---|---:|---|---|" % ("---:|---:|" * len(allp)))
        for r in ex:
            note = ""
            if r.get("best_ever_params_differ") or r.get("best_ever_hash_differs"):
                note = "vs best ever: " + ",".join(r.get("best_ever_params_differ", [])[:6] +
                                                   (["hash"] if r.get("best_ever_hash_differs") else []))
            elif r.get("best_ever_ms") is None:
                note = "no earlier measurement"
            xvals = "".join("%s | %s | " % (fmt_ms(((r.get("priors") or {}).get(x) or {}).get("ms")),
                                            fmt_ratio(((r.get("priors") or {}).get(x) or {}).get("ratio")))
                            for x in allp)
            lines.append("| %s | %s | %s%s%s | %s |" % (r["key"], fmt_ms(r.get("new_ms")), xvals, best_ever_cells(r),
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
    # the verdict is printed with BOTH arms' probes beside it: a verdict without the probe is not a verdict (AGENTS.md)
    out.append("verdict: " + str(res.get("verdict_basis")))
    out.append("  probe candidate: " + probe_line(res.get("host_probe")))
    for p in res.get("prior_columns") or []:
        out.append("  probe %s: %s%s" % (p["version"], probe_line(p.get("probe")),
                                         " [%s]" % p["why_not"] if p.get("why_not") else ""))
    out.append("  stored %s (reference, no flags): host %s%s" % (
        res["paper_version"], res.get("stored_host") or "not recorded",
        "; %s" % res["reference_label"] if res.get("reference_label") else ""))
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
    for flag in ("RECOVERED", "STILL-REGRESSED", "FIXED", "BROKEN", "BASELINE-NOT-RUN", "NOT-COMPARED",
                 "QUALITY-DROP", "QUALITY-CHANGE",
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
    priors = []
    for spec in args.prior_board or []:
        version, path = parse_prior_board(spec)
        try:
            priors.append((version, load_json(path)))
        except (OSError, ValueError) as exc:
            # a prior that never ran (its install or every smoke failed) is a missing column, not a crash
            print("PAPER-COMPARE unreadable prior board %s: %s (column NOT-RUN)" % (path, exc), file=sys.stderr)
            priors.append((version, {"races": {}}))
    ref, rlabel = (None, None)
    if args.reference_compare:
        ref, rlabel = load_reference(args.reference_compare)
    best = None
    if args.best_ever:
        best = BE.load(os.path.join(REPO, "bench", "results", "board_best_ever", "%s.json" % doc["vendor"])
                       if args.best_ever == "auto" else args.best_ever)
    res = compare(doc, boards, why, priors, ref, rlabel, best)
    v = doc["vendor"]
    md = args.out_md or os.path.join(OUT_DIR, "COMPARE-%s.md" % v)
    js = args.out_json or os.path.join(os.path.dirname(os.path.abspath(md)), "compare-%s.json" % v)
    summary = render_summary(res)
    for path, text in ((md, render_md(res)), (js, json.dumps(res, indent=1, sort_keys=True) + "\n"),
                       (args.summary or os.path.join(os.path.dirname(os.path.abspath(md)), "SUMMARY-%s.txt" % v),
                        summary + "\n")):
        write_atomic(path, text)
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
    c.add_argument("--prior-board", action="append", default=[], metavar="VERSION=BOARD.json",
                   help="OUR prior wheel of VERSION re-timed in the same job on the same box (board-prior-<v>/"
                        "board.json); repeat. The paper's version is the baseline (REGRESSION / FASTER / FIXED / "
                        "BROKEN), the last other one the prior column (REGRESSION-VS-PRIOR, RECOVERED, "
                        "STILL-REGRESSED). A board from another host refuses")
    c.add_argument("--reference-compare", default=None,
                   help="a stored compare JSON of another run (bench/results/paper_compare/compare-<vendor>.json): "
                        "a reference column with its host, never a flag")
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
