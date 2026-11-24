#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn
"""tools/ops/targeted_rows.py -- the TARGETED measurement (Andrew, 2026-11-17: "can we only measure the things that
have changed? we don't need a full board run"): pick the board rows worth timing, and the box-side helpers of
tools/ops/paper_compare_job.sh PC_PARTS=targeted, which times only those rows on the candidate and on 0.8.55 on the
same box in the same job and prints the full board's compare for them.

SELECT (the Mac, seconds; writes the rows file the job reads as PC_ROWS):

    python3 tools/ops/targeted_rows.py --from-compare <compare-all json> [--slower 1.10] [--changed-since <sha>]
        [--board <board.json>] [--prior-board <board.json>] [--out <rows.tsv>] [--cap-min 230] [--chunk-s 600]

  A row (a race id of the board's current plan, tools/paper_compare.py plan_all) is selected for each reason:
    slower         any of its compare rows (recipe, extra or internal) is slower than --slower against the on-box
                   0.8.55 column of the given compare (never the stored or best-ever columns: those never flag);
    paper          a paper recipe row flagged REGRESSION in the given compare;
    changed        its lane's source closure changed between --changed-since and HEAD: the lane's public estimator
                   (the race's recorded call, else the lane's ours_class) -> the module that defines it
                   (mojolearn/__init__.py _LAZY_NAMES) -> the bindings that module names (`_mojolearn_<x>`) -> each
                   binding's build-script closure digest from tools/release_reuse.py facts_for_commit (the build
                   reuse's own identity code: a binding whose digest differs between the two commits is changed),
                   plus the module file itself. The Python side stops at the defining module: every module reaches
                   all 127 package files through the package's shared imports, so a deeper walk would select every
                   row for any Python edit. A race whose mapping is uncertain (no recorded call, no module, no binding
                   with a build script, a lane the stored board never ran) is included ("changed?");
    internal       the one internal call-overhead row (tools/race_fairness.py INTERNAL_CALL_OVERHEAD_LANE).
  Rows go in that order (paper, slower worst first, changed, internal): the job times them in file order, so a cap
  kill loses the least wanted rows. It prints the count per reason, rows the compare had that the board no longer
  plans, and the estimated pod minutes for both arms from each row's stored wall seconds in the last board.json and
  its 0.8.55 board (a row with no stored time: the median of its family). Over --cap-min it writes the split
  (<out>.part<i>.tsv, each under the cap, each a complete same-box job) and names it; tools/ops/measure_candidate.py
  stage --rows refuses a rows file over the cap with the same split.

OWN-ONLY (Andrew, 2026-11-17: "we only run A/B and what has changed"; 0.8.55 is not timed again):

    python3 tools/ops/targeted_rows.py --own-only --from-compare <compare-all json> [--results-dir DIR]
        [--merged-into REF] [--measured SHA] [--control N] [--seed S] [--out <rows.tsv>]

  selects ONLY the races a merged fix covers, read from the lanes' per-row result files (RESULT_GLOBS in
  --results-dir): an entry selects its races when its fix cell names a commit that is on --merged-into and not already
  in the measured build --measured (the 15cca53af1 class-layout rows included where a lane assigned them). Not
  selected: entries classed harness (class b), noise or box state, no cause found / not fixed / needs a timing split,
  waiting on an A/B only (timed by its A/B arm); a back-to-back "not a regression" vetoes the race whatever fix another
  entry names. Plus the internal call-overhead row and --control N seeded random SAME/FASTER races (default 0). It
  prints per fix commit how many races it selects, every entry NOT selected with its reason, the entries that name no
  planned race (also in <out>.audit.txt), and the one-arm estimate (our new wheel only; the stored baseline is
  compared in tools/paper_compare.py --stored-baseline-board).

  The rows file: `# ` comment lines, then one row per line, tab separated:
      <race id>  <reasons, comma list>  <est candidate s>  <est 0.8.55 s>  <input ratio, the worst of the race's rows
                                                                           against 0.8.55 in the given compare, or ->

BOX HELPERS (tools/ops/paper_compare_job.sh, CPU, seconds):
    plan     --rows F --vendor V --planner P --recipes R --out-dir D: the listed races the board plans now (rows-file
             order) with their smoke ids, board args, the recipes of those races only (the compare's paper rows),
             races.json (the datasets), not-planned.json, and an empty paper doc for tools/measure_owed.py plan (the
             owed B units are timed with their own base here, so none is left to a paper board); prints the estimate.
    chunks   --ids F --rows R --chunk-s N --out-dir D: the interleave chunks (see the job): consecutive races whose
             estimated candidate seconds add up to about N; chunk-<i>.txt.
    prep-groups --ids F [--ids G] --out-dir D: the races cut into one group per family and prep dataset
             (group-<family>-<dataset>.txt), which the job's tg-prep phase prepares at once, one process each.
    timed-check --results RES --ids F [--units U] [--arms A,B]: one UNTIMED line per staged arm (the candidate on the
             listed rows, each owed unit's arm and its base) that has no timed race, with why; exit 3 when any. A unit
             timed on both sides whose arm and base hold different rows gets one PARTIAL line naming them (no exit change).
    ab-gaps  --arm A --rows T --ids F --base B --board X: one AB-ROWS-DIFFER line naming the races done on one side of
             an A/B unit and not the other, with the reason (the preflight record, the race status, the cell status).
    empty-board --out D [--host-probe F] [--why W]: D/board.json with no race and the job's host probe, written
             only when D has no board (the candidate timed nothing), so the compare names the host and the NOT-RUN rows.
    summary  --compare C --rows R --results RES --ids F [--units U]: SUMMARY-TG (rows timed on both,
             FASTER/SAME/REGRESSION, rows still over 1.10 worst first, rows fixed (over 1.10 in the input compare, now
             at or under 1.00), done/total per arm, and each owed A/B unit's arm over its back-to-back base: the median
             verdict, then the direction reading (cells faster/flat/slower on the raw ratio, min and max, ONE-WAY), and
             every cell or race without a time with its reason).

PARALLEL PLAN (HARD RULE, 2026-11-07): the two commits' closure facts (git archive + the Mojo import walk, about two
minutes each) are computed at once in two processes; everything else is a pass over JSON in one process (seconds,
nothing worth a core). On the box the timed races are serial by nature (one process on the GPU); the job's installs,
smokes and compare use every cgroup core.
"""
import argparse
import ast
import json
import os
import re
import statistics
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "ops"))
import paper_compare as PC  # noqa: E402
import race_fairness as RF  # noqa: E402

REASONS = ("paper", "slower", "changed", "changed?", "fix", "internal", "control")
#: the job's fixed cost outside the timed races, seconds (d95f805cce full board, L40S, job/phases.tsv: probe 2, plan
#: 0, install 8-46, compare 1): rounded up
FIXED_S = 90
#: untimed smoke seconds per race and arm. The d95f805cce all-smoke phase ran 650 candidate + 627 0.8.55 races in
#: 805 s (0.65 s a race) on a pod with a larger cgroup; the targeted A/B job ab1 of d652a4b0a1 (L40S pod
#: 37f71a6102af, 14-core cgroup, so 7 smoke shards) ran 212 races in 759 s (3.6 s a race) against an estimate of
#: 2 minutes (lane fb15-ab-verdicts, 2026-11-18). The shard count is half the cgroup's cores, so the 14-core
#: figure is the one to plan with on the current pods.
SMOKE_S_PER_RACE = 3.6
#: the start of one timed bench_board process (interpreter, plan, preflight audit, smoke gate) per chunk and arm,
#: with the prepared data already in the cache (the tg-prep phase of tools/ops/paper_compare_job.sh). Measured
#: (lane fx-pod-job, 2026-11-18, from ideas-cycle-time-RESULT.md 1d-1e): the A/B job nvc-ab-20261118T140734Z-8f0269
#: spent 1215 s in the starts of 77 bench_board processes (16 s each, the median); the 30 s of ab1 included the prep
#: now done once in PREP_S.
#: Re-measured (lane ab-harness-speed, 2026-11-24, the two A/B jobs of 4519097d380a on a 14-core L40S pod): tg-board
#: less its race walls and install windows, over its bench_board processes: (6769 - 6115 - 41) / 45 = 13.6 s and
#: (6913 - 6030 - 129) / 75 = 10.1 s; 12 s
PROC_S = 12
#: the job's dataset preparation, ONCE per job (tg-prep: one process per dataset, all at once, before the smoke):
#: the longest single dataset's prep. Measured in nvc-tg-20261118T135756Z-7a2bff: graph-taxi 256.9 s and
#: graph-istella 229.5 s, then serial inside the first timed process; in parallel the job waits for the longest.
#: TOMBSTONE (lane fx-pod-job, 2026-11-18): RACE_GAP_S = 10 (seconds between two timed races on top of their stored
#: wall_s) is removed: the same job's sum of race wall_s (1776 s) equals the sum of the races' "done" seconds
#: (1789 s), so there is no gap; the 9.6 s a race of ab1 was the process starts and the serial prep, counted above.
#: With it the A/B job of 2026-11-18 was estimated at 224 min and took 101, and the planner split jobs that fit.
PREP_S = 260
#: a stored wall_s from another pod is a floor, not a forecast: the races whose own time is host-bound (the quality
#: references on the CPU) scale with the cgroup. ab1 took elliptic-envelope istella 1748 s against a stored 510 s
#: (28-core pod) and hdbscan istella 726 s against 164 s. tools/ops/measure_candidate.py therefore estimates each
#: race from the LARGEST stored wall_s over every board it is given (the 14-core ab1 boards among them).
DEFAULT_CAP_MIN = 230       # under tools/nvidia_central.sh's 240-minute cap, with the job's own teardown
DEFAULT_CHUNK_S = 600
DEFAULT_SEED = 20261117
BINDING_RE = re.compile(r"_mojolearn_[a-z0-9_]+")
CALL_RE = re.compile(r"^\s*(mojolearn(?:\.[A-Za-z_][A-Za-z0-9_]*)+)")


# ---------------------------------------------------------------------------------------------------------------
# the rows file
# ---------------------------------------------------------------------------------------------------------------
def read_rows(path):
    """[{id, reasons, new_s, prior_s, input_ratio}] in file order; a bare id line is a row with no estimate."""
    out = []
    with open(path) as fh:
        for ln in fh:
            ln = ln.rstrip("\n")
            if not ln.strip() or ln.lstrip().startswith("#"):
                continue
            f = ln.split("\t")
            num = lambda i: float(f[i]) if len(f) > i and f[i] not in ("", "-") else None  # noqa: E731
            out.append({"id": f[0].strip(), "reasons": [r for r in (f[1] if len(f) > 1 else "").split(",") if r],
                        "new_s": num(2), "prior_s": num(3), "input_ratio": num(4)})
    return out


def rows_text(rows, header=()):
    lines = ["# %s\n" % h for h in header]
    lines.append("# race id\treasons\test candidate s\test 0.8.55 s\tinput ratio vs 0.8.55\n")
    for r in rows:
        lines.append("%s\t%s\t%s\t%s\t%s\n" % (
            r["id"], ",".join(r["reasons"]), "%.1f" % r["new_s"] if r.get("new_s") is not None else "-",
            "%.1f" % r["prior_s"] if r.get("prior_s") is not None else "-",
            "%.3f" % r["input_ratio"] if r.get("input_ratio") is not None else "-"))
    return "".join(lines)


# ---------------------------------------------------------------------------------------------------------------
# the estimate and the split
# ---------------------------------------------------------------------------------------------------------------
def estimate(rows, chunk_s=DEFAULT_CHUNK_S, one_arm=False, prior_wheels=0):
    """{candidate_s, prior_s, wheels_s, smoke_s, overhead_s, chunks, total_s, total_min} of a job over ROWS. ONE_ARM:
    the job times no 0.8.55 arm (the stored baseline; Andrew, 2026-11-17). PRIOR_WHEELS: how many of our own earlier
    built wheels (measure_candidate.py --prior-wheel) the job times on every row beside the candidate; each is
    estimated at the candidate's stored seconds (the same code base a few commits back), with its smoke and its
    process per chunk."""
    if one_arm:
        rows = [dict(r, prior_s=None) for r in rows]
    new = sum(r.get("new_s") or 0.0 for r in rows)
    prior = sum(r.get("prior_s") or 0.0 for r in rows)
    wheels = prior_wheels * new
    nchunks = len(chunk_ids([r["id"] for r in rows], {r["id"]: r for r in rows}, chunk_s))
    smoke = SMOKE_S_PER_RACE * ((1 + prior_wheels) * len(rows) + sum(1 for r in rows if (r.get("prior_s") or 0) > 0))
    arms = (1 if one_arm else 2) + prior_wheels
    over = FIXED_S + PREP_S + arms * PROC_S * nchunks
    total = new + prior + wheels + smoke + over
    return {"candidate_s": new, "prior_s": prior, "wheels_s": wheels, "prior_wheels": prior_wheels, "smoke_s": smoke,
            "overhead_s": over, "chunks": nchunks, "total_s": total, "total_min": total / 60.0, "one_arm": one_arm}


def estimate_line(est):
    return ("TG-ESTIMATE minutes total=%.0f candidate=%.0f 0.8.55=%s%s smoke=%.0f overhead=%.0f (chunks=%d, %d s per "
            "process, %d s of prep once)" % (est["total_min"], est["candidate_s"] / 60,
                                             ("stored (screen)" if est.get("prior_wheels") else "stored (one arm)")
                                             if est.get("one_arm") else "%.0f" % (est["prior_s"] / 60),
                                             " prior-wheels=%.0f (%d)" % (est["wheels_s"] / 60, est["prior_wheels"])
                                             if est.get("prior_wheels") else "",
                                             est["smoke_s"] / 60, est["overhead_s"] / 60, est["chunks"], PROC_S,
                                             PREP_S))


def split_rows(rows, cap_min, chunk_s=DEFAULT_CHUNK_S, one_arm=False, prior_wheels=0):
    """ROWS in file order cut into parts whose estimate is under CAP_MIN each (a row alone over the cap is a part of
    its own: it cannot be split further)."""
    parts, cur = [], []
    for r in rows:
        if cur and estimate(cur + [r], chunk_s, one_arm, prior_wheels)["total_min"] > cap_min:
            parts.append(cur)
            cur = []
        cur.append(r)
    if cur:
        parts.append(cur)
    return parts


def check_cap(rows_path, cap_min, chunk_s=DEFAULT_CHUNK_S, write_split=True, one_arm=False, prior_wheels=0):
    """(estimate, [split file paths]) of a rows file; the split list is empty when it fits under CAP_MIN. The parts
    are written beside it as <stem>.part<i>.tsv, each one complete job (every arm it times on the same box)."""
    rows = read_rows(rows_path)
    est = estimate(rows, chunk_s, one_arm, prior_wheels)
    if est["total_min"] <= cap_min:
        return est, []
    p = Path(rows_path)
    files = []
    for i, part in enumerate(split_rows(rows, cap_min, chunk_s, one_arm, prior_wheels), 1):
        f = p.with_name("%s.part%d%s" % (p.stem, i, p.suffix or ".tsv"))
        e = estimate(part, chunk_s, one_arm, prior_wheels)
        if write_split:
            PC.write_atomic(str(f), rows_text(part, ["part %d of %s (cap %s min): %s" % (
                i, p.name, cap_min, estimate_line(e))]))
        files.append((str(f), e))
    return est, files


def chunk_ids(ids, by_id, chunk_s):
    """IDS in order, cut where the estimated candidate seconds of a chunk reach CHUNK_S (one race per chunk when
    CHUNK_S is 0). A race without an estimate counts CHUNK_S / 4."""
    chunks, cur, acc = [], [], 0.0
    for i in ids:
        s = (by_id.get(i) or {}).get("new_s")
        s = chunk_s / 4.0 if s is None else s
        if cur and acc + s > chunk_s:
            chunks.append(cur)
            cur, acc = [], 0.0
        cur.append(i)
        acc += s
    if cur:
        chunks.append(cur)
    return chunks


# ---------------------------------------------------------------------------------------------------------------
# select
# ---------------------------------------------------------------------------------------------------------------
def compare_rows(doc):
    """[(race id, ratio vs the on-box paper baseline or None, row, kind)] of every recipe, extra and internal row
    (kind: recipe / extra / internal). A recipe row's ratio counts only with a verdict flag (an on-box baseline). The
    baseline is the compare's own (baseline_version: a prior wheel under --verdict-prior), else the paper's version."""
    pv = doc.get("baseline_version") or doc.get("paper_version") or PC.PAPER_VERSION
    out = []
    for r in doc.get("rows") or []:
        verdict = any(f in ("REGRESSION", "FASTER", "SAME") for f in r.get("flags") or [])
        out.append((r["race_id"], r.get("ratio") if verdict else None, r, "recipe"))
    for kind in ("extra", "internal"):
        for r in doc.get(kind + "_rows") or []:
            p = (r.get("priors") or {}).get(pv) or {}
            out.append((r["race_id"], p.get("ratio"), r, kind))
    return out


def lazy_names(pkg):
    """mojolearn/__init__.py _LAZY_NAMES: {public name: '.module' or None (a submodule)}."""
    tree = ast.parse((Path(pkg) / "__init__.py").read_text())
    for n in tree.body:
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "_LAZY_NAMES":
            return ast.literal_eval(n.value)
    return {}


def module_file(pkg, dotted):
    """The file of a package module ('.x.y' or 'x.y'), or None."""
    rel = [p for p in dotted.lstrip(".").split(".") if p]
    base = Path(pkg).joinpath(*rel) if rel else Path(pkg)
    if (base / "__init__.py").is_file():
        return base / "__init__.py"
    f = base.with_suffix(".py")
    return f if f.is_file() else None


def race_module(race, lazy, pkg):
    """(module file relative to the repo, why not) of a stored board race: its recorded call (`mojolearn.X(...)`),
    else its lane's ours_class."""
    names = []
    for c in (race.get("cells") or []) + (race.get("infer_cells") or []):
        call = c.get("call") if str(c.get("arm", "")).startswith("ours") else None
        if isinstance(call, str):
            names.append(call)
    lc = race.get("lane_config") if isinstance(race.get("lane_config"), dict) else {}
    if isinstance(lc.get("ours_class"), str):
        names.append(lc["ours_class"])
    for text in names:
        m = CALL_RE.match(text)
        if not m:
            continue
        parts = m.group(1).split(".")[1:]
        if len(parts) == 1 and parts[0] in lazy:
            mod = lazy[parts[0]] or "." + parts[0]
        elif parts and parts[0] in lazy and lazy[parts[0]] is None:
            mod = "." + parts[0]                        # mojolearn.<submodule>.<name>
        else:
            continue
        f = module_file(pkg, mod)
        if f is not None:
            return f, None
    return None, "no public estimator recorded" if not names else "no module for %s" % names[0][:60]


def module_bindings(path):
    return sorted(set(BINDING_RE.findall(Path(path).read_text(errors="replace"))))


def script_of(binding):
    return "build.sh" if binding == "_mojolearn" else "build_" + binding[len("_mojolearn_"):] + ".sh"


def _facts(args):
    commit, cache = args
    import release_reuse as RR
    return RR.facts_for_commit(commit, cache)


def closure_change(since, head, cache_dir, root=ROOT):
    """({build script: changed?}, [changed files]) between two commits, from tools/release_reuse.py's facts (the
    build reuse's closure digests): both commits' facts at once (two processes)."""
    with ProcessPoolExecutor(2) as ex:
        old, new = list(ex.map(_facts, [(since, cache_dir), (head, cache_dir)]))
    oc, nc = old.get("closures") or {}, new.get("closures") or {}
    changed = {s: (oc.get(s) or {}).get("digest") != (nc.get(s) or {}).get("digest") or nc.get(s) is None
               for s in set(oc) | set(nc)}
    files = subprocess.run(["git", "-C", str(root), "diff", "--name-only", since, head], capture_output=True,
                           text=True, check=True).stdout.split()
    return changed, files, nc


def changed_races(plan_ids, board, since, head="HEAD", cache_dir=None, root=ROOT):
    """({race id: 'changed' | 'changed?'}, notes) for the planned races whose lane closure changed (module docstring)."""
    head_sha = subprocess.run(["git", "-C", str(root), "rev-parse", head], capture_output=True, text=True,
                              check=True).stdout.strip()
    cache = cache_dir or str(Path.home() / "mojolearn-evidence" / "targeted" / "facts-cache")
    scripts, files, closures = closure_change(since, head_sha, cache, root)
    files = set(files)
    pkg = root / "python" / "mojolearn"
    lazy = lazy_names(pkg)
    races = (board or {}).get("races") or {}
    out, why_uncertain, mod_cache = {}, {}, {}
    for rid in plan_ids:
        race = races.get(rid)
        if race is None:
            out[rid] = "changed?"
            why_uncertain[rid] = "the stored board never ran it"
            continue
        f, why = race_module(race, lazy, pkg)
        if f is None:
            out[rid] = "changed?"
            why_uncertain[rid] = why
            continue
        if f not in mod_cache:
            binds = [b for b in module_bindings(f) if script_of(b) in closures]
            mod_cache[f] = binds
        binds = mod_cache[f]
        if not binds:
            out[rid] = "changed?"
            why_uncertain[rid] = "%s names no binding with a build script" % f.relative_to(root)
            continue
        if str(f.relative_to(root)) in files or any(scripts.get(script_of(b)) for b in binds):
            out[rid] = "changed"
    n_changed = sum(1 for s, c in scripts.items() if c)
    shared = {}
    for s, c in scripts.items():
        for src in (closures.get(s) or {}).get("sources") or []:
            if src in files:
                shared[src] = shared.get(src, 0) + 1
    widest = sorted(shared.items(), key=lambda kv: -kv[1])[:5]
    notes = ["closures changed %d of %d build scripts (tools/release_reuse.py digests %s..%s)"
             % (n_changed, len(scripts), since[:10], head_sha[:10]),
             "widest changed closure files: " + ", ".join("%s (%d scripts)" % kv for kv in widest)]
    if why_uncertain:
        kinds = {}
        for w in why_uncertain.values():
            k = re.sub(r" for .*| names no .*", "", w)
            kinds[k] = kinds.get(k, 0) + 1
        notes.append("uncertain mapping (included): " + ", ".join("%s %d" % kv for kv in sorted(kinds.items())))
    return out, notes


# ---------------------------------------------------------------------------------------------------------------
# --own-only: the rows a merged fix covers (Andrew, 2026-11-17: "we only run A/B and what has changed")
# ---------------------------------------------------------------------------------------------------------------
#: the lanes' per-row result files (the source of truth for --own-only): TSVs with a row column and a fix-like
#: column, and markdown reports whose tables name rows and their fix or status
RESULT_DIR = Path.home() / "mojolearn-evidence" / "briefs-20261117"
RESULT_GLOBS = ("fb-*-RESULT.tsv", "fb2-*-RESULT.tsv", "fb3-band-RESULT.tsv", "fb3-named-RESULT.md",
                "fb4-ten-RESULT.md")
INTEGRATION_REF = "origin/preinteg/wave2-20261113"
SHA_RE = re.compile(r"\b[0-9a-f]{10,40}\b")
#: the column (lower-cased header) a TSV entry's fix is read from, first match wins; a row's cause column is never
#: read for shas (it names the commit that CAUSED the regression, fb-neural: "6869626579 ... _graph_key")
FIX_COLUMNS = ("fix", "fix or timing split", "fix_or_split", "explanation", "status")
#: markdown cells read for a fix (the cause / delta cells are not)
MD_FIX_HEADERS = ("fix", "status", "finding", "what reading established", "explanation")
#: an entry whose fix text starts like this names no fix of its own, whatever sha it mentions in passing ("not found
#: (the cgroup cap 9fd0275a2d may cover it)": a guess, not an assignment)
NO_FIX_START = re.compile(r"^\s*(not found|not fixed|needs timing split|on-box split|re-time|open\b|-\s*$|$)", re.I)
#: the build the stored baseline compare measured (d95f805cce, the 2026-11-17 L40S full board): a commit already in it
#: changed nothing since that measurement, so it selects no row (the lanes cite it as "the product path at d95f805cce")
MEASURED_BUILD = "d95f805cce"
NOT_REGRESSION = re.compile(r"not a regression|not reproducible|shows no regression", re.I)
EXCLUDE_KINDS = (
    ("harness", re.compile(r"\bharness\b", re.I)),
    ("noise or box state", re.compile(r"noise|box[ -]state|drift|not a regression|not reproducible|"
                                      r"single call, no structure", re.I)),
    ("waiting on an A/B only (timed by its A/B arm)", re.compile(r"owed a/b|a/b owed", re.I)),
)


def _cells_md(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def result_entries(paths):
    """[{file, row, cls, fix, text}] of every per-row entry in the lanes' result files. TSV: the header's row column
    (first), class column when there is one, and the fix column (FIX_COLUMNS, else the last). Markdown: every table
    row; fix = the cells under MD_FIX_HEADERS (else the last cell)."""
    out = []
    for p in paths:
        name = os.path.basename(str(p))
        lines = Path(p).read_text(errors="replace").splitlines()
        if str(p).endswith(".tsv"):
            if not lines:
                continue
            head = [h.strip().lower() for h in lines[0].split("\t")]
            if head[0] not in ("row",):
                continue                                    # per-site tables (fb-boost-copy, fb5-cpython-layout)
            fi = next((head.index(c) for c in FIX_COLUMNS if c in head), len(head) - 1)
            ci = head.index("class") if "class" in head else None
            for ln in lines[1:]:
                f = ln.split("\t")
                if not f or not f[0].strip():
                    continue
                out.append({"file": name, "row": f[0].strip(), "cls": f[ci].strip() if ci is not None and ci < len(f)
                            else "", "fix": f[fi].strip() if fi < len(f) else "", "text": ln})
            continue
        head = None
        for ln in lines:
            if not ln.lstrip().startswith("|"):
                head = None
                continue
            cells = _cells_md(ln)
            if all(set(c) <= set("-: ") for c in cells):
                continue
            if head is None:
                head = [c.lower() for c in cells]
                continue
            # the fix / status cell when the table has one; else the finding cell (a cause cell names culprits)
            idx = ([i for i, h in enumerate(head) if h in ("fix", "status")]
                   or [i for i, h in enumerate(head) if h in MD_FIX_HEADERS] or [len(cells) - 1])
            out.append({"file": name, "row": cells[0], "cls": "",
                        "fix": " ".join(cells[i] for i in idx if i < len(cells)), "text": ln})
    return out


def is_merged(sha, ref=INTEGRATION_REF, root=ROOT, cache={}):  # noqa: B006 (a per-process memo)
    key = (sha, ref)
    if key not in cache:
        cache[key] = subprocess.run(["git", "-C", str(root), "merge-base", "--is-ancestor", sha, ref],
                                    capture_output=True).returncode == 0
    return cache[key]


def entry_decision(e, merged=is_merged):
    """('fix', [merged shas]) or ('excluded', why) of one entry (module docstring, --own-only)."""
    fix, cls = e["fix"], e["cls"].lower()
    if cls.startswith("b") and "a" not in cls:
        return "excluded", "harness (class %s)" % e["cls"]
    # a fix cell that names a commit assigns the row to it even when it also says "harness already fixed" or "plus
    # box state" (fb2-smallcall, fb4-ten: the 15cca53af1 rows); without one, those words exclude the row
    if not SHA_RE.search(fix) or NO_FIX_START.match(fix):
        for why, rx in EXCLUDE_KINDS[:2]:
            if rx.search(e["text"]):
                return "excluded", why
    if NO_FIX_START.match(fix):
        if EXCLUDE_KINDS[2][1].search(e["text"]):
            return "excluded", EXCLUDE_KINDS[2][0]
        return "excluded", "no cause found or not fixed (%s)" % (fix[:60] or "-")
    shas = list(dict.fromkeys(SHA_RE.findall(fix)))
    if not shas:
        if EXCLUDE_KINDS[2][1].search(fix):
            return "excluded", EXCLUDE_KINDS[2][0]
        return "excluded", "names no fix commit (%s)" % fix[:60]
    ok = [s for s in shas if merged(s)]
    if not ok:
        return "excluded", "fix %s not on %s, or already in the measured build %s" % (
            ",".join(shas), INTEGRATION_REF, MEASURED_BUILD)
    return "fix", ok


def race_index(plan_ids):
    """({lane: [race ids]}, {datasets}) of the board's current plan."""
    lanes, ds = {}, set()
    for rid in plan_ids:
        p = rid.split("/")
        if len(p) >= 3:
            lanes.setdefault(p[1], []).append(rid)
            ds.add(p[2])
    return lanes, ds


def resolve_row(text, plan_ids, index=None):
    """The planned race ids an entry's row text names: a race id or a key ('fam/lane/ds/rows=full fit ours'), a
    'fam/lane/ds phase', a 'lane/ds phase', or prose ('kernel-shap taxi / istella', 'gcn / graphsage fit (4 rows)':
    every planned lane named, on the datasets named, else on all of its datasets)."""
    planned = set(plan_ids)
    lanes, dsets = index or race_index(plan_ids)
    first = text.split()[0] if text.split() else ""
    seg = first.split("/")
    if len(seg) >= 4 and "/".join(seg[:4]) in planned:
        return ["/".join(seg[:4])]
    if len(seg) == 3 and seg[1] in lanes:
        return [r for r in lanes[seg[1]] if r.split("/")[0] == seg[0] and r.split("/")[2] == seg[2]]
    if len(seg) == 2 and seg[0] in lanes and seg[1] in dsets:
        return [r for r in lanes[seg[0]] if r.split("/")[2] == seg[1]]
    toks = [t for t in re.split(r"[\s,/()]+", text) if t]
    named = [t for t in toks if t in lanes]
    want = {t for t in toks if t in dsets}
    out = []
    for lane in dict.fromkeys(named):
        out += [r for r in lanes[lane] if not want or r.split("/")[2] in want]
    return out


def own_only_select(plan_ids, entries, merged=is_merged):
    """({race id: [merged fix shas]}, [(entry, why)] not selected, [entries naming no planned race]): a race is
    selected when ANY entry naming it names a merged fix (an entry classed harness / noise / box state names none)."""
    index = race_index(plan_ids)
    fixes, rejected, unmapped = {}, [], []
    for e in entries:
        rids = resolve_row(e["row"], plan_ids, index)
        kind, why = entry_decision(e, merged)
        if not rids:
            unmapped.append((e, why if kind == "excluded" else "fix " + ",".join(why)))
            continue
        e["races"] = rids
        if kind == "fix":
            for rid in rids:
                fixes.setdefault(rid, [])
                fixes[rid] += [s for s in why if s not in fixes[rid]]
        else:
            rejected.append((e, why))
    # a back-to-back attribution that showed the row is NOT a regression vetoes it, whatever fix another entry names
    # (fb4-ten: gbdt-symmetric infer "Not a regression"; fb3-named: gbdt-depthwise / lossguide "not reproducible")
    for e, why in rejected:
        if why == "noise or box state" and NOT_REGRESSION.search(e["text"]):
            for r in e["races"]:
                fixes.pop(r, None)
            e["veto"] = True
    not_sel = [(e, why + (" (vetoes any fix)" if e.get("veto") else "")) for e, why in rejected
               if not any(r in fixes for r in e["races"])]
    return fixes, not_sel, unmapped


def stored_seconds(board):
    return {rid: r.get("wall_s") for rid, r in ((board or {}).get("races") or {}).items()
            if r.get("status") == "done" and isinstance(r.get("wall_s"), (int, float))}


def family_medians(secs):
    fam = {}
    for rid, s in secs.items():
        fam.setdefault(rid.split("/")[0], []).append(s)
    return {f: statistics.median(v) for f, v in fam.items()}


def select(compare_doc, plan_ids, slower, board, prior_board, changed=None):
    """(rows in job order, counts per reason, [compare race ids the board no longer plans], estimated-count)."""
    planned = set(plan_ids)
    worst, paper = {}, set()
    gone = set()
    for rid, ratio, row, kind in compare_rows(compare_doc):
        if rid not in planned:
            gone.add(rid)
            continue
        if ratio is not None and (worst.get(rid) is None or ratio > worst[rid]):
            worst[rid] = ratio
        if kind == "recipe" and "REGRESSION" in (row.get("flags") or []):
            paper.add(rid)
    reasons = {}
    for rid in paper:
        reasons.setdefault(rid, []).append("paper")
    for rid, ratio in worst.items():
        if ratio is not None and ratio > slower:
            reasons.setdefault(rid, []).append("slower")
    for rid, why in (changed or {}).items():
        if rid in planned:
            reasons.setdefault(rid, []).append(why)
    for rid in plan_ids:
        if rid.split("/")[1] == RF.INTERNAL_CALL_OVERHEAD_LANE:
            reasons.setdefault(rid, []).append("internal")
    new_s, prior_s = stored_seconds(board), stored_seconds(prior_board)
    med = family_medians(new_s)
    rows, guessed = [], 0
    for rid, rs in reasons.items():
        n = new_s.get(rid)
        if n is None:
            n = med.get(rid.split("/")[0])
            guessed += 1
        rows.append({"id": rid, "reasons": rs, "new_s": n, "prior_s": prior_s.get(rid, 0.0),
                     "input_ratio": worst.get(rid)})
    order = {r: i for i, r in enumerate(REASONS)}

    def key(r):
        first = min(order.get(x, 9) for x in r["reasons"])
        return (first, -(r["input_ratio"] or 0.0), r["id"])
    rows.sort(key=key)
    counts = {r: sum(1 for x in rows if r in x["reasons"]) for r in REASONS}
    return rows, counts, sorted(gone), guessed


def own_only_rows(plan_ids, fixes, compare_doc, board, control=0, seed=DEFAULT_SEED):
    """(rows in job order: the fix rows (most fixes first, then worst input ratio), the internal row, then CONTROL
    seeded random races that were SAME or FASTER in the stored compare (a check that nothing fine got slower), all
    with no 0.8.55 estimate (one arm), counts per reason, how many have no stored time)."""
    import random
    planned = set(plan_ids)
    worst, verdict = {}, {}
    for rid, ratio, row, _kind in compare_rows(compare_doc):
        if rid in planned and ratio is not None:
            worst[rid] = max(worst.get(rid, ratio), ratio)
            verdict[rid] = verdict.get(rid, True) and ratio <= PC.SLOWER
    reasons = {rid: ["fix"] + ["fix=" + s for s in shas] for rid, shas in fixes.items() if rid in planned}
    for rid in plan_ids:
        if rid.split("/")[1] == RF.INTERNAL_CALL_OVERHEAD_LANE:
            reasons.setdefault(rid, []).append("internal")
    pool = sorted(rid for rid, ok in verdict.items() if ok and rid not in reasons)
    for rid in random.Random(seed).sample(pool, min(control, len(pool))):
        reasons[rid] = ["control"]
    new_s = stored_seconds(board)
    med = family_medians(new_s)
    rows, guessed = [], 0
    for rid, rs in reasons.items():
        n = new_s.get(rid)
        if n is None:
            n = med.get(rid.split("/")[0])
            guessed += 1
        rows.append({"id": rid, "reasons": rs, "new_s": n, "prior_s": None, "input_ratio": worst.get(rid)})
    order = {"fix": 0, "internal": 1, "control": 2}
    rows.sort(key=lambda r: (order[r["reasons"][0]], -len(r["reasons"]), -(r["input_ratio"] or 0.0), r["id"]))
    counts = {k: sum(1 for x in rows if k in x["reasons"]) for k in ("fix", "internal", "control")}
    return rows, counts, guessed


def cmd_own_only(a, planner, plan_ids):
    doc = PC.load_json(a.from_compare)
    d = os.path.dirname(os.path.abspath(a.from_compare))
    board = PC.load_json(a.board or os.path.join(d, "board-all", "board.json"))
    paths = sorted({str(p) for g in RESULT_GLOBS for p in Path(a.results_dir).glob(g)})
    entries = result_entries(paths)
    # a fix counts when it is on the integration branch and NOT already in the measured build
    merged = lambda s: is_merged(s, a.merged_into) and not is_merged(s, a.measured)  # noqa: E731
    fixes, not_sel, unmapped = own_only_select(plan_ids, entries, merged)
    rows, counts, guessed = own_only_rows(plan_ids, fixes, doc, board, a.control, a.seed)
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short=12", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    out = a.out or str(Path.home() / "mojolearn-evidence" / "targeted" / ("rows-own-%s.tsv" % head))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    per_fix = {}
    for rid, shas in fixes.items():
        if rid in set(plan_ids):
            for s in shas:
                per_fix[s] = per_fix.get(s, 0) + 1
    audit = ["own-only (tools/ops/targeted_rows.py --own-only) at %s: %d result files in %s, %d entries, fixes merged "
             "into %s" % (head, len(paths), a.results_dir, len(entries), a.merged_into)]
    audit += ["fix %s selects %d races" % kv for kv in sorted(per_fix.items(), key=lambda kv: (-kv[1], kv[0]))]
    PC.write_atomic(out, rows_text(rows, audit))
    lines = list(audit)
    lines.append("NOT selected (%d entries; a race another entry assigns to a merged fix is selected, not listed):"
                 % len(not_sel))
    lines += ["  %s [%s] %s: %s" % (e["row"][:90], e["file"], ",".join(e["races"])[:160], why) for e, why in not_sel]
    lines.append("no planned race (%d entries: not on the board now, a paper or prose row, or a note line):"
                 % len(unmapped))
    lines += ["  %s [%s]: %s" % (e["row"][:90], e["file"], why) for e, why in unmapped]
    PC.write_atomic(out + ".audit.txt", "\n".join(lines) + "\n")
    print("TG-ROWS %s rows=%d planned_now=%d own-only %s" % (out, len(rows), len(plan_ids),
                                                         " ".join("%s=%d" % kv for kv in counts.items())))
    for ln in lines:
        print("TG-OWN " + ln)
    if guessed:
        print("TG-NOTE %d rows have no stored candidate time: family median used" % guessed)
    est, split = check_cap(out, a.cap_min, a.chunk_s, one_arm=True)
    print(estimate_line(est))
    if split:
        print("TG-SPLIT over the %s-minute cap: %d parts, each a complete job:" % (a.cap_min, len(split)))
        for f, e in split:
            print("  %s  %s" % (f, estimate_line(e)))
    return 0


def cmd_select(a):
    planner = PC.load_planner(a.planner)
    plan_ids = PC.plan_all(planner, a.vendor)[0]
    if a.own_only:
        return cmd_own_only(a, planner, plan_ids)
    doc = PC.load_json(a.from_compare)
    d = os.path.dirname(os.path.abspath(a.from_compare))
    board = PC.load_json(a.board or os.path.join(d, "board-all", "board.json"))
    prior = PC.load_json(a.prior_board or os.path.join(d, "board-all-prior-%s" % PC.PAPER_VERSION, "board.json"))
    changed, notes = ({}, [])
    if a.changed_since:
        changed, notes = changed_races(plan_ids, board, a.changed_since, cache_dir=a.facts_cache)
    rows, counts, gone, guessed = select(doc, plan_ids, a.slower, board, prior, changed)
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short=12", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    out = a.out or str(Path.home() / "mojolearn-evidence" / "targeted" / ("rows-%s.tsv" % head))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    header = ["targeted rows (tools/ops/targeted_rows.py) at %s: --from-compare %s --slower %s --changed-since %s"
              % (head, a.from_compare, a.slower, a.changed_since or "-")] + notes
    PC.write_atomic(out, rows_text(rows, header))
    est, split = check_cap(out, a.cap_min, a.chunk_s)
    print("TG-ROWS %s rows=%d planned_now=%d %s" % (out, len(rows), len(plan_ids),
                                                     " ".join("%s=%d" % (k, v) for k, v in counts.items())))
    for n in notes:
        print("TG-NOTE " + n)
    if gone:
        print("TG-GONE %d compare races the board no longer plans (not selected): %s" % (
            len(gone), ", ".join(gone[:6]) + (" ..." if len(gone) > 6 else "")))
    if guessed:
        print("TG-NOTE %d rows have no stored candidate time: family median used" % guessed)
    print(estimate_line(est))
    if split:
        print("TG-SPLIT over the %s-minute cap: %d parts, each a complete job:" % (a.cap_min, len(split)))
        for f, e in split:
            print("  %s  %s" % (f, estimate_line(e)))
    return 0


# ---------------------------------------------------------------------------------------------------------------
# box helpers
# ---------------------------------------------------------------------------------------------------------------
def cmd_plan(a):
    rows = read_rows(a.rows)
    planner = PC.load_planner(a.planner)
    full, smoke, missing, bargs, races = PC.plan_all(planner, a.vendor)
    sid = dict(zip(full, smoke))
    meta = {r["race_id"]: r for r in races}
    want, seen = [], set()
    for r in rows:
        if r["id"] not in seen:
            seen.add(r["id"])
            want.append(r["id"])
    ids = [i for i in want if i in sid]
    not_planned = {i: missing.get(i, "the board does not plan this race on %s now" % a.vendor)
                   for i in want if i not in sid}
    os.makedirs(a.out_dir, exist_ok=True)
    w = lambda name, text: PC.write_atomic(os.path.join(a.out_dir, name), text)  # noqa: E731
    w("race-ids-full.txt", "".join(i + "\n" for i in ids))
    w("race-ids-smoke.txt", "".join(sid[i] + "\n" for i in ids))
    w("board-args.txt", "".join(x + "\n" for x in bargs))
    w("not-planned.json", PC.json_text(not_planned))
    w("races.json", PC.json_text({"recipes": [meta[i] for i in ids]}))
    rec = PC.load_json(a.recipes)
    keep = set(ids)
    rec = dict(rec, recipes=[r for r in rec.get("recipes") or [] if r.get("race_id") in keep],
               race_ids=[i for i in rec.get("race_ids") or [] if i in keep])
    w("recipes.json", PC.json_text(rec))
    # tools/measure_owed.py plan's paper doc: no race is left to a paper board, so every owed B unit gets its own base
    w("owed-paper.json", PC.json_text({"recipes": [], "race_ids": [], "bench_board_args": {}}))
    by_id = {r["id"]: r for r in rows}
    print("TG-PLAN vendor=%s rows=%d planned=%d not_planned=%d paper_recipes=%d" % (
        a.vendor, len(want), len(ids), len(not_planned), len(rec["recipes"])))
    print(estimate_line(estimate([by_id[i] for i in ids], a.chunk_s, a.one_arm, a.prior_wheels)))
    return 0


def cmd_chunks(a):
    with open(a.ids) as fh:
        ids = [x.strip() for x in fh if x.strip()]
    by_id = {r["id"]: r for r in read_rows(a.rows)} if a.rows else {}
    chunks = chunk_ids(ids, by_id, a.chunk_s)
    os.makedirs(a.out_dir, exist_ok=True)
    for f in Path(a.out_dir).glob("chunk-*.txt"):
        f.unlink()
    for i, c in enumerate(chunks, 1):
        PC.write_atomic(os.path.join(a.out_dir, "chunk-%03d.txt" % i), "".join(x + "\n" for x in c))
    print("TG-CHUNKS races=%d chunks=%d chunk_s=%d" % (len(ids), len(chunks), a.chunk_s))
    return 0


#: the datasets each family's block prep is keyed by (tools/bench_board.py DATASETS): a race on any other dataset
#: (synthetic, text, ...) preps under the first of them, so it joins that group and no two groups write one block
PREP_DATASETS = ("taxi", "istella")


def prep_groups(ids, datasets=PREP_DATASETS):
    """{group name: [race id]} of IDS ('family/lane/dataset/...'), one group per family and prep dataset, in first
    appearance order. Each family preps into its own cache dir (algos-data, more-data, ctd-data) and its blocks are
    keyed by dataset, so the groups write disjoint files and run at once (tools/ops/paper_compare_job.sh tg-prep)."""
    out = {}
    for i in ids:
        f = i.split("/")
        fam = f[0] if f and f[0] else "x"
        ds = f[2] if len(f) > 2 and f[2] in datasets else datasets[0]
        out.setdefault("%s-%s" % (fam, ds), []).append(i)
    return out


def is_full(race_id):
    """A timed (full-size) race id: its size field is rows=full or shape=full. A smoke id (rows=2000, shape=small)
    is planned only by a --rows/--smoke board, never by the timed board's plan."""
    return race_id.rsplit("/", 1)[-1].endswith("=full")


def cmd_prep_groups(a):
    ids, dropped = [], 0
    for path in a.ids:
        if os.path.isfile(path):
            with open(path) as fh:
                for x in fh:
                    x = x.strip()
                    if not x or x in ids:
                        continue
                    # lane fx-pod-job2 (2026-11-18): the smoke ids reached this list, the full-row plan refused the
                    # whole group ("--race-ids: not in this plan") and nothing was prepared
                    if not is_full(x):
                        dropped += 1
                        continue
                    ids.append(x)
    if dropped:
        print("TG-PREP-GROUPS dropped %d smoke-size id(s): only full-size races are prepared here" % dropped)
    groups = prep_groups(ids)
    os.makedirs(a.out_dir, exist_ok=True)
    for f in Path(a.out_dir).glob("group-*.txt"):
        f.unlink()
    for name, gids in groups.items():
        PC.write_atomic(os.path.join(a.out_dir, "group-%s.txt" % name), "".join(x + "\n" for x in gids))
    print("TG-PREP-GROUPS races=%d groups=%d (%s)" % (len(ids), len(groups),
                                                     ", ".join("%s %d" % (k, len(v)) for k, v in groups.items())))
    return 0


def _lines(path):
    try:
        with open(path) as fh:
            return [x.strip() for x in fh if x.strip()]
    except OSError:
        return None


def _first(path, prefix=None, last=False):
    """The first line starting with PREFIX (or the last non-empty line), cut to 240 characters; None without one."""
    try:
        with open(path, errors="replace") as fh:
            lines = [x.strip() for x in fh if x.strip()]
    except OSError:
        return None
    if prefix is not None:
        lines = [x[len(prefix):].strip() for x in lines if x.startswith(prefix)]
    if not lines:
        return None
    return (lines[-1] if last else lines[0])[:240]


def _statuses(board_path, ids):
    try:
        races = PC.load_json(board_path).get("races") or {}
    except (OSError, ValueError):
        return None
    out = {}
    for i in ids:
        st = (races.get(i) or {}).get("status") or "absent"
        out[st] = out.get(st, 0) + 1
    return ", ".join("%s %d" % kv for kv in sorted(out.items()))


def untimed(results, ids, units_path=None, only_arms=None, prior_wheels=()):
    """[(arm, why)]: every staged arm of a targeted job that has no timed race (lane fx-pod-job2, 2026-11-18: 12 of
    14 B arms were skipped with a quiet "no race to time" and the job's only failure was a missing board). The
    candidate counts when rows are listed (board-tg); each owed unit counts with its arm board and its base board
    (board-tg-ab-<arm>[-base]-<rows>). The why names the cause: no race passed the smoke (its first failure), no
    board (the install or board log's last line), or the races' statuses."""
    out = []
    logs = os.path.join(results, "logs")
    if ids:
        b = os.path.join(results, "board-tg", "board.json")
        if _done(b, ids) == 0:
            passed = _lines(os.path.join(results, "plan-tg", "race-ids-pass.txt"))
            if passed is not None and not passed:
                why = "none of its %d listed race(s) passed the smoke: %s" % (
                    len(ids), _first(os.path.join(logs, "pass-ids-tg.log"), "PAPER-PASS-IDS-FAILED") or "?")
            else:
                why = "no listed race done in board-tg (%s; logs/board-tg.log: %s)" % (
                    _statuses(b, ids) or "no board", _first(os.path.join(logs, "board-tg.log"), last=True) or "-")
            out.append(("candidate", why))
    for name in prior_wheels or ():
        # a prior wheel (PC_PRIOR_WHEELS) is the verdict baseline: with no timed listed race there is no verdict
        arm = "prior-" + name
        b = os.path.join(results, "board-tg-" + arm, "board.json")
        if ids and _done(b, ids) == 0:
            passed = _lines(os.path.join(results, "plan-tg", arm, "race-ids-pass.txt"))
            if passed is not None and not passed:
                why = "none of the listed races passed both smokes: %s" % (
                    _first(os.path.join(logs, "pass-ids-tg-%s.log" % arm), "PAPER-PASS-IDS-FAILED") or "?")
            else:
                why = "no listed race done in board-tg-%s (%s; logs/install-tg-%s.log: %s; logs/board-tg-%s.log: %s)" % (
                    arm, _statuses(b, ids) or "no board", arm,
                    _first(os.path.join(logs, "install-tg-%s.log" % arm), last=True) or "-", arm,
                    _first(os.path.join(logs, "board-tg-%s.log" % arm), last=True) or "-")
            out.append((arm, why))
    if not units_path or not os.path.isfile(units_path):
        return out
    with open(units_path) as fh:
        units = [ln.rstrip("\n").split("\t") for ln in fh]
    for f in units:
        if len(f) != 3 or f[0] in ("candidate", "prior"):
            continue
        arm, rows, ids_file = f
        if only_arms and arm not in only_arms:
            continue
        uids = _lines(ids_file) or []
        if not uids:
            continue
        name = "board-tg-ab-%s-%s" % (arm, rows)
        ab = os.path.join(results, name, "board.json")
        use = _lines(ids_file + ".pass") if rows == "full" else uids
        if use is not None and not use:
            out.append((arm, "%s rows: none of its %d race(s) passed the smoke: %s" % (
                rows, len(uids), _first(os.path.join(logs, "pass-ids-tg-ab-%s.log" % arm), "PAPER-PASS-IDS-FAILED")
                or "?")))
            continue
        use = use or uids
        if _done(ab, use) == 0:
            if not os.path.isfile(ab):
                why = "no board %s (logs/install-tg-%s.log: %s; logs/%s.log: %s)" % (
                    name, arm, _first(os.path.join(logs, "install-tg-%s.log" % arm), last=True) or "-",
                    name, _first(os.path.join(logs, name + ".log"), last=True) or "-")
            else:
                why = "no race done in %s (%s; logs/%s.log: %s)" % (
                    name, _statuses(ab, use), name, _first(os.path.join(logs, name + ".log"), last=True) or "-")
            out.append((arm, "%s rows: %s" % (rows, why)))
            continue
        base_ids = _lines(ids_file + (".pass.base" if rows == "full" else ".base"))
        bname = "board-tg-ab-%s-base-%s" % (arm, rows)
        bb = os.path.join(results, bname, "board.json")
        if base_ids and _done(bb, base_ids) == 0:
            out.append((arm, "%s rows: its base (the candidate) has no race done in %s (%s; logs/%s.log: %s)" % (
                rows, bname, _statuses(bb, base_ids) or "no board", bname,
                _first(os.path.join(logs, bname + ".log"), last=True) or "-")))
    return out


def partial(results, units_path=None, only_arms=None):
    """[(arm, AB-ROWS-DIFFER line)] of every owed unit timed on both sides whose arm and base boards hold different
    done races (lane fx-tools-round3, 2026-11-20: gemm-leaf512's arm board held 3 of its base's 4 rows, the unit ended
    rc=1 and untimed() said nothing, because the arm had SOME timed race). A reading, not an UNTIMED arm."""
    import measure_owed as MO
    out = []
    if not units_path or not os.path.isfile(units_path):
        return out
    with open(units_path) as fh:
        units = [ln.rstrip("\n").split("\t") for ln in fh]
    for f in units:
        if len(f) != 3 or f[0] in ("candidate", "prior") or (only_arms and f[0] not in only_arms):
            continue
        arm, rows, ids_file = f
        use = (_lines(ids_file + ".pass") if rows == "full" else None) or _lines(ids_file) or []
        base = MO.load_board(os.path.join(results, "board-tg-ab-%s-base-%s" % (arm, rows), "board.json"))
        brd = MO.load_board(os.path.join(results, "board-tg-ab-%s-%s" % (arm, rows), "board.json"))
        if base is None or brd is None or not use:
            continue     # an untimed side is untimed()'s line
        line = gap_line(arm, rows, ab_row_gaps(base, brd, use))
        if line:
            out.append((arm, line))
    return out


def cmd_timed_check(a):
    ids = _lines(a.ids) or []
    only = [x for x in (a.arms or "").split(",") if x] or None
    bad = untimed(a.results, ids, a.units, only, [x for x in (a.prior_wheels or "").split(",") if x])
    for arm, why in bad:
        print("UNTIMED %s: %s" % (arm, why))
    for arm, line in partial(a.results, a.units, only):
        print("PARTIAL %s: %s" % (arm, line))
    print("TG-TIMED-CHECK untimed=%d%s" % (len(bad), "" if not bad else " (the job exits non-zero)"))
    return 3 if bad else 0


def cmd_empty_board(a):
    """The candidate's targeted board when no race of it was timed (lane fx-pod-job2, 2026-11-18: tg-compare read a
    missing board-tg/board.json and printed 'this job: host not recorded'): races {} and the job's host probe, so the
    compare shows every listed row NOT-RUN with its reason and names this job's host. A timed board never has
    'placeholder'; the job deletes this file before the first timed process writes the dir."""
    path = os.path.join(a.out, "board.json")
    if os.path.isfile(path):
        print("TG-EMPTY-BOARD kept %s (it exists)" % path)
        return 0
    probe = None
    if a.host_probe:
        try:
            probe = PC.load_json(a.host_probe)
        except (OSError, ValueError):
            probe = None
    doc = {"schema": "mojolearn-bench-board/1", "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "placeholder": a.why or "no race of the candidate was timed", "box": {"host_probe": probe}, "races": {}}
    os.makedirs(a.out, exist_ok=True)
    PC.write_atomic(path, json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print("TG-EMPTY-BOARD wrote %s (%s)" % (path, doc["placeholder"]))
    return 0


def _done(board_path, ids):
    try:
        races = (PC.load_json(board_path).get("races") or {})
    except (OSError, ValueError):
        races = {}
    return sum(1 for i in ids if (races.get(i) or {}).get("status") == "done")


#: the A/B direction reading (lane fx-tools-round3, 2026-11-20: tf32-all had 12 of 13 cells faster and none slower
#: and read "median 0.93 SAME"): a cell under 0.99 is faster, over 1.01 slower, between them flat; ONE-WAY when at least
#: 80% of the cells (and at least 5) move one way with none beyond 1.05 the other way. An added reading only: the
#: per-cell FASTER/SLOWER flags and the median verdict keep PC.FASTER / PC.SLOWER.
DIR_FLAT = 0.01
ONE_WAY_SHARE = 0.80
ONE_WAY_MIN = 5
ONE_WAY_AGAINST = 0.05


def direction(ratios):
    """'faster F flat N slower S min X max Y[ ONE-WAY faster|slower]' of an arm's raw cell ratios ('' with none)."""
    if not ratios:
        return ""
    lo, hi = 1.0 - DIR_FLAT, 1.0 + DIR_FLAT
    f = sum(1 for r in ratios if r < lo)
    s = sum(1 for r in ratios if r > hi)
    text = "faster %d flat %d slower %d min %s max %s" % (f, len(ratios) - f - s, s, PC.fmt_ratio(min(ratios)),
                                                          PC.fmt_ratio(max(ratios)))
    n = len(ratios)
    for way, k, against in (("faster", f, any(r > 1.0 + ONE_WAY_AGAINST for r in ratios)),
                            ("slower", s, any(r < 1.0 - ONE_WAY_AGAINST for r in ratios))):
        if k >= ONE_WAY_MIN and k >= ONE_WAY_SHARE * n and not against:
            text += " ONE-WAY " + way
    return text


def missing_reason(brd, rid, key=None):
    """Why race RID (and, with KEY, its our-arm cell (phase, arm, batch)) has no timed cell in board document BRD:
    no board, the preflight record (failed_preflight_recipes / skipped_recipes with its problems), the race not in the
    board, the race's status and failure, or the cell's own status. Never empty (lane fx-tools-round3, 2026-11-20:
    gemm-leaf512's arm lost neural/gemm to the profile admission and its line read 'arm - ms ratio -')."""
    import measure_owed as MO
    if brd is None:
        return "no board"
    for field, word in (("failed_preflight_recipes", "FAILED"), ("skipped_recipes", "REFUSED")):
        rec = (brd.get(field) or {}).get(rid)
        if rec is not None:
            probs = [str(p) for p in (rec.get("problems") or [])] if isinstance(rec, dict) else [str(rec)]
            return "preflight %s: %s" % ((rec.get("status") if isinstance(rec, dict) else None) or word,
                                         "; ".join(probs[:3])[:300] or "no problem recorded")
    race = (brd.get("races") or {}).get(rid)
    if race is None:
        return "race not in the board (never reached: see the board's log)"
    why = "race %s" % (race.get("status") or "without status")
    if race.get("failure"):
        why += ": " + str(race["failure"])[:240]
    if key is not None:
        cell = MO.our_cells(race).get(key)
        if cell is None:
            return why + "; no %s %s cell" % (key[0], key[1])
        if race.get("status") in (None, "done"):
            return "cell %s" % PC.short_status(cell.get("status"))[:240]
        why += "; cell %s" % PC.short_status(cell.get("status"))[:160]
    return why


def ab_row_gaps(base, brd, uids):
    """[(race id, side, why)] of the unit's races done on one side and not on the other (side 'arm' or 'base'):
    a row an arm lost is named with its reason, never only a shorter count (lane fx-tools-round3, 2026-11-20)."""
    out = []
    for rid in uids:
        b_done = ((((base or {}).get("races") or {}).get(rid)) or {}).get("status") == "done"
        a_done = ((((brd or {}).get("races") or {}).get(rid)) or {}).get("status") == "done"
        if b_done and not a_done:
            out.append((rid, "arm", missing_reason(brd, rid)))
        elif a_done and not b_done:
            out.append((rid, "base", missing_reason(base, rid)))
    return out


def gap_line(arm, rows_tag, gaps):
    """The one AB-ROWS-DIFFER line of an A/B unit whose arm and base hold different rows ('' when they match)."""
    if not gaps:
        return ""
    return "AB-ROWS-DIFFER arm=%s rows=%s %s" % (arm, rows_tag, " | ".join(
        "%s missing on the %s: %s" % (rid, side, why) for rid, side, why in gaps))


def cmd_ab_gaps(a):
    """Print the AB-ROWS-DIFFER line of one unit (the job calls it after each unit; exit 0 always: a reading)."""
    import measure_owed as MO
    uids = _lines(a.ids) or []
    line = gap_line(a.arm, a.rows, ab_row_gaps(MO.load_board(a.base), MO.load_board(a.board), uids))
    print(line or "AB-ROWS-SAME arm=%s rows=%s (%d race(s))" % (a.arm, a.rows, len(uids)))
    return 0


def summary(doc, rows, results, ids, units=(), prior_versions=None, loop_noise=None, loop_out=None):
    """The SUMMARY-TG text (module docstring). Against a stored baseline (tools/paper_compare.py
    --stored-baseline-board) its first line is the compare's own: the baseline is stored, with both sides' host, GPU,
    driver, kernel and date; BOX-STATE rows (the upload probe moved) are counted apart and never as a verdict."""
    import measure_owed as MO
    want = {r["id"] for r in rows}
    input_ratio = {r["id"]: r.get("input_ratio") for r in rows}
    control = {r["id"] for r in rows if "control" in r.get("reasons", [])}
    # the verdict baseline: the paper's version, or a prior wheel timed in this job (compare --verdict-prior)
    pv = doc.get("baseline_version") or doc.get("paper_version") or PC.PAPER_VERSION
    stored = bool(doc.get("baseline_stored"))
    lines = []
    if stored:
        lines += [doc.get("verdict_basis") or "STORED BASELINE"] + list(doc.get("baseline_warnings") or [])
    elif doc.get("screen_stored"):
        lines.append("verdict: %s; the stored %s board is a screen column only (no flag)" % (
            doc.get("verdict_basis"), doc.get("paper_version")))
    lines.append("TARGETED vendor=%s %s -> %s rows=%d (%s)" % (
        doc.get("vendor"), pv, doc.get("new_label"), len(rows),
        "baseline stored, our new wheel timed alone" if stored else (doc.get("verdict_basis") or "no verdict")))
    counts, over, now, box, slower_ctrl, seen = {}, [], {}, [], [], set()
    for rid, _ratio, r, kind in compare_rows(doc):
        if rid not in want:
            continue
        flags = [f for f in r.get("flags") or [] if f in ("FASTER", "SAME", "REGRESSION", "BOX-STATE")]
        ratio = r.get("ratio") if kind == "recipe" else ((r.get("priors") or {}).get(pv) or {}).get("ratio")
        if ratio is None or not flags:
            continue
        counts[flags[0]] = counts.get(flags[0], 0) + 1
        old = r.get("old_ms") if kind == "recipe" else ((r.get("priors") or {}).get(pv) or {}).get("ms")
        if flags[0] == "BOX-STATE":
            box.append((ratio, r.get("key") or rid, r.get("upload_probe")))
            seen.add(rid)
            continue
        now[rid] = max(now.get(rid, ratio), ratio)
        seen.add(rid)
        if ratio > PC.SLOWER:
            over.append((ratio, r.get("key") or rid, r.get("new_ms"), old))
            if rid in control:
                slower_ctrl.append((ratio, r.get("key") or rid))
    lines.append("timed=%d FASTER=%d SAME=%d REGRESSION=%d BOX-STATE=%d (rows of %d races; BOX-STATE: the in-race "
                 "upload probe moved more than %d%% from the stored run's, ratio shown, not counted)" % (
                     sum(counts.values()), counts.get("FASTER", 0), counts.get("SAME", 0),
                     counts.get("REGRESSION", 0), counts.get("BOX-STATE", 0), len(seen),
                     round(100 * PC.BOX_STATE_TOLERANCE)))
    lines.append("still over %.2f (%d), worst first:" % (PC.SLOWER, len(over)))
    for ratio, key, new, old in sorted(over, key=lambda x: -x[0]):
        lines.append("  %.2fx %s (%s -> %s ms)" % (ratio, key, PC.fmt_ms(old), PC.fmt_ms(new)))
    fixed = sorted(rid for rid, r in input_ratio.items() if r is not None and r > PC.SLOWER
                   and now.get(rid) is not None and now[rid] <= 1.0)
    lines.append("fixed (over %.2f in the input compare, now at or under 1.00): %d" % (PC.SLOWER, len(fixed)))
    lines += ["  %s (%.2fx -> %.2fx)" % (rid, input_ratio[rid], now[rid]) for rid in fixed]
    if control:
        lines.append("control rows that got slower (over %.2f): %d of %d control races" % (
            PC.SLOWER, len(slower_ctrl), len(control)))
        lines += ["  %.2fx %s" % x for x in sorted(slower_ctrl, reverse=True)]
    if box:
        lines.append("BOX-STATE rows (%d), ratio not counted:" % len(box))
        lines += ["  %.2fx %s; probe %s" % (ratio, key, PC.probe_text(p)) for ratio, key, p in sorted(box, reverse=True)]
    lines.append("done/total candidate %d/%d" % (_done(os.path.join(results, "board-tg", "board.json"), ids), len(ids)))
    if stored or doc.get("screen_stored"):
        lines.append("%s: stored, not timed in this job" % doc.get("paper_version"))
    for v in ([] if stored else (prior_versions or [pv])):
        pid = os.path.join(results, "plan-tg", "prior-%s" % v, "race-ids-pass.txt")
        lines.append("done/total %s %d/%d (of the listed races; %s passed its own smoke)" % (
            v, _done(os.path.join(results, "board-tg-prior-%s" % v, "board.json"), ids), len(ids),
            sum(1 for _ in open(pid)) if os.path.isfile(pid) else "?"))
    import ab_loop as AL
    noise = AL.NOISE if loop_noise is None else float(loop_noise)
    table, detail, loop_lines, loop_doc = [], [], [], {"schema": AL.SCHEMA, "noise": noise, "units": []}
    for arm, rows_tag, uids, capped_ids in units:
        loop_pairs = []
        base = MO.load_board(os.path.join(results, "board-tg-ab-%s-base-%s" % (arm, rows_tag), "board.json"))
        brd = MO.load_board(os.path.join(results, "board-tg-ab-%s-%s" % (arm, rows_tag), "board.json"))
        ratios, untimed_cells, start = [], 0, len(detail)
        for rid in uids:
            bc = MO.our_cells(((base or {}).get("races") or {}).get(rid) or {})
            ac = MO.our_cells(((brd or {}).get("races") or {}).get(rid) or {})
            keys = sorted(set(bc) | set(ac), key=str)
            if not keys:
                # a race neither board timed: its line names why instead of vanishing (lane fx-tools-round3)
                untimed_cells += 1
                detail.append("AB %s rows=%s %s: no cell on either side (base: %s; arm: %s)" % (
                    arm, rows_tag, rid, missing_reason(base, rid), missing_reason(brd, rid)))
                continue
            for k in keys:
                b, x = MO.cell_ms(bc.get(k)), MO.cell_ms(ac.get(k))
                r = x / b if b and x else None
                lp = AL.pair(bc.get(k), ac.get(k), noise) if b and x else None
                if b and x:
                    loop_pairs.append((rid, bc.get(k), ac.get(k)))
                if r is not None:
                    ratios.append(r)
                else:
                    untimed_cells += 1
                if r is None and b and rid in capped_ids and MO.capped(ac.get(k)):
                    # IN-JOB BASE CAP (measure_owed.py): the arm's rounds stopped at this job's base time
                    v = " NOT-FASTER-THAN-BASE (stopped at this job's base time)"
                elif r is None:
                    # the side without a time says why (lane fx-tools-round3, 2026-11-20)
                    v = " NOT-TIMED (%s)" % "; ".join(
                        "%s: %s" % (side, missing_reason(d, rid, k))
                        for side, d, ms in (("base", base, b), ("arm", brd, x)) if not ms)
                else:
                    v = " " + MO.verdict(r, "FASTER", "SLOWER", "SAME")
                # A/B LOOP MODE: the loop per iteration the verdict reads, or why the cell is the whole operation.
                # REMOVED (lane ab-loop-mode, 2026-11-24): per_iteration_text, the whole operation over the executed
                # iterations with its PHASE-VERDICTS-DIFFER mark and count (lane ab-harness-speed's interim reading);
                # the product's loop region now decides (tools/ab_loop.py), so that proxy has no decision left
                ph = AL.pair_text(lp) if lp is not None else ""
                detail.append("AB %s rows=%s %s %s %s: base %s -> arm %s ms ratio %s%s%s%s" % (
                    arm, rows_tag, rid, k[0], k[1], PC.fmt_ms(b), PC.fmt_ms(x), PC.fmt_ratio(r), v, ph,
                    base_used(base, brd, rid)))
        med = statistics.median(ratios) if ratios else None
        gaps = ab_row_gaps(base, brd, uids)
        table.append("  %-34s rows=%-6s races=%-3d cells=%-3d median arm/base %s %s%s%s%s" % (
            arm, rows_tag, len(uids), len(ratios), PC.fmt_ratio(med),
            "-" if med is None else MO.verdict(med, "FASTER", "SLOWER", "SAME"),
            ("; " + direction(ratios)) if ratios else "",
            ("; untimed cells %d" % untimed_cells) if untimed_cells else "",
            ("; rows differ %d" % len(gaps)) if gaps else ""))
        if gaps:
            detail.insert(start, gap_line(arm, rows_tag, gaps))
        # A/B LOOP MODE (tools/ab_loop.py): the arm's verdict on loop time per iteration, under every condition
        v, _ = AL.unit_verdict(loop_pairs, uids, noise)
        loop_lines.append(AL.verdict_line(arm, rows_tag, v))
        loop_doc["units"].append(dict(v, arm=arm, rows=rows_tag))
    if table:
        lines.append("A/B (arm and its base, the candidate, back to back):")
        lines += table + detail
    if loop_lines:
        lines.append("A/B LOOP MODE verdicts (AGENTS.md Candidate evaluation; noise floor %.2f; FULL-RUN-OWED arms "
                     "go to the next submit's PC_AB_LOOP_FULL_ARMS):" % noise)
        lines += loop_lines
        owed = [u["arm"] for u in loop_doc["units"] if u.get("full_run")]
        if owed:
            lines.append("AB-LOOP-FULL-RUN-OWED PC_AB_LOOP_FULL_ARMS=%s" % ",".join(sorted(set(owed))))
    if loop_out:
        PC.write_atomic(loop_out, PC.json_text(loop_doc))
    return "\n".join(lines) + "\n"


def base_used(base, brd, rid):
    """'; base <board> gap <s> s': which base run the arm's race was compared with (its own unit's base, or the
    SHARED BASE another unit timed, tools/measure_owed.py base-share) and the seconds from that base race's end to the
    arm race's start; '' when either side has no time stamp."""
    import measure_owed as MO
    b =(((base or {}).get("races") or {}).get(rid)) or {}
    a = (((brd or {}).get("races") or {}).get(rid)) or {}
    sh = b.get("shared_base") or {}
    src = os.path.basename(str(sh.get("board"))) if sh else "own"
    end, start = MO._iso_s(sh.get("finished") or b.get("finished")), MO._iso_s(a.get("started"))
    if end is None or start is None:
        return "; base %s" % src if b else ""
    return "; base %s gap %d s" % (src, int(start - end))


def read_units(path):
    """[(B arm, rows tag, [ids], {capped ids})] of a tools/measure_owed.py units.tsv (the candidate units are the bases;
    a unit's ids-<arm>-<rows>.cap.txt lists the races whose arm rounds were capped at this job's base time)."""
    out = []
    if not path or not os.path.isfile(path):
        return out
    with open(path) as fh:
        for ln in fh:
            f = ln.rstrip("\n").split("\t")
            if len(f) == 3 and f[0] not in ("candidate", "prior") and not os.path.isfile(f[2]):
                # its ids file is gone: the unit stays in the table as races=0, never dropped (lane fx-tools-round3)
                out.append((f[0], f[1], [], set()))
            elif len(f) == 3 and f[0] not in ("candidate", "prior"):
                with open(f[2]) as ih:
                    ids = [x.strip() for x in ih if x.strip()]
                cf = os.path.splitext(f[2])[0] + ".cap.txt"
                caps = set(x.strip() for x in open(cf) if x.strip()) if os.path.isfile(cf) else set()
                out.append((f[0], f[1], ids, caps))
    return out


def cmd_summary(a):
    doc = PC.load_json(a.compare)
    with open(a.ids) as fh:
        ids = [x.strip() for x in fh if x.strip()]
    text = summary(doc, read_rows(a.rows), a.results, ids, read_units(a.units),
                   [v for v in (a.prior_versions or "").split(",") if v] or None, a.loop_noise, a.loop_out)
    if a.out:
        PC.write_atomic(a.out, text)
    sys.stdout.write(text)
    return 0


def build_select_parser():
    p = argparse.ArgumentParser(prog="targeted_rows.py", description=__doc__.split("\n\n")[0])
    p.add_argument("--from-compare", required=True, help="a compare-all JSON (tools/paper_compare.py compare)")
    p.add_argument("--slower", type=float, default=PC.SLOWER, help="new / 0.8.55 above this selects a row")
    p.add_argument("--changed-since", default=None, help="a commit: rows whose lane closure changed since it")
    p.add_argument("--board", default=None, help="the candidate board.json (default <compare dir>/board-all)")
    p.add_argument("--prior-board", default=None,
                   help="the 0.8.55 board.json (default <compare dir>/board-all-prior-%s)" % PC.PAPER_VERSION)
    p.add_argument("--vendor", default="nvidia", choices=PC.VENDORS)
    p.add_argument("--planner", default=str(ROOT / "tools" / "bench_board.py"))
    p.add_argument("--out", default=None, help="the rows file (default ~/mojolearn-evidence/targeted/rows-<head>.tsv)")
    p.add_argument("--cap-min", type=float, default=DEFAULT_CAP_MIN)
    p.add_argument("--chunk-s", type=int, default=DEFAULT_CHUNK_S)
    p.add_argument("--facts-cache", default=None, help="tools/release_reuse.py facts cache dir")
    p.add_argument("--own-only", action="store_true",
                   help="the rows a MERGED fix covers (the lanes' per-row result files, --results-dir) plus the "
                        "internal call-overhead row, timed on our new wheel only against the stored baseline")
    p.add_argument("--results-dir", default=str(RESULT_DIR), help="the lanes' *-RESULT.tsv / .md files")
    p.add_argument("--merged-into", default=INTEGRATION_REF, help="a fix counts when it is an ancestor of this ref")
    p.add_argument("--measured", default=MEASURED_BUILD,
                   help="the build the stored compare measured: a fix already in it selects nothing")
    p.add_argument("--control", type=int, default=0,
                   help="--own-only: add N seeded random races that were SAME or FASTER in --from-compare")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p


def build_box_parser():
    p = argparse.ArgumentParser(prog="targeted_rows.py")
    sub = p.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("plan")
    pl.add_argument("--rows", required=True)
    pl.add_argument("--vendor", required=True, choices=PC.VENDORS)
    pl.add_argument("--planner", default=str(ROOT / "tools" / "bench_board.py"))
    pl.add_argument("--recipes", required=True)
    pl.add_argument("--out-dir", required=True)
    pl.add_argument("--chunk-s", type=int, default=DEFAULT_CHUNK_S)
    pl.add_argument("--one-arm", action="store_true", help="the stored-baseline job: no 0.8.55 arm")
    pl.add_argument("--prior-wheels", type=int, default=0,
                    help="how many of our own earlier built wheels (PC_PRIOR_WHEELS) the job times beside the candidate")
    ch = sub.add_parser("chunks")
    ch.add_argument("--ids", required=True)
    ch.add_argument("--rows", default=None)
    ch.add_argument("--chunk-s", type=int, default=DEFAULT_CHUNK_S)
    ch.add_argument("--out-dir", required=True)
    pg = sub.add_parser("prep-groups", help="the listed races cut into the groups tg-prep prepares at once")
    pg.add_argument("--ids", action="append", required=True, help="an ids file (repeat it; a missing file is skipped)")
    pg.add_argument("--out-dir", required=True)
    tc = sub.add_parser("timed-check", help="every staged arm timed? one UNTIMED line per arm that was not; exit 3")
    tc.add_argument("--results", required=True)
    tc.add_argument("--ids", required=True, help="the listed races (plan-tg/race-ids-full.txt)")
    tc.add_argument("--units", default=None, help="the owed plan's units.tsv")
    tc.add_argument("--arms", default=None, help="only these arms (a split submit's PC_OWED_ARMS)")
    tc.add_argument("--prior-wheels", default=None,
                    help="comma list of the prior wheels (PC_PRIOR_WHEELS names): each must have a timed listed race")
    eb = sub.add_parser("empty-board", help="board-tg/board.json with no race, when the candidate timed none")
    eb.add_argument("--out", required=True)
    eb.add_argument("--host-probe", default=None)
    eb.add_argument("--why", default=None)
    ag = sub.add_parser("ab-gaps", help="one AB-ROWS-DIFFER line: the races an A/B unit holds on one side only, why")
    ag.add_argument("--arm", required=True)
    ag.add_argument("--rows", required=True, help="the unit's rows tag")
    ag.add_argument("--ids", required=True, help="the unit's ids file (both sides time the same list)")
    ag.add_argument("--base", required=True, help="the base board.json")
    ag.add_argument("--board", required=True, help="the arm board.json")
    sm = sub.add_parser("summary")
    sm.add_argument("--compare", required=True)
    sm.add_argument("--rows", required=True)
    sm.add_argument("--results", required=True)
    sm.add_argument("--ids", required=True, help="the listed races the board plans (plan-tg/race-ids-full.txt)")
    sm.add_argument("--units", default=None, help="the owed plan's units.tsv (the A/B arms)")
    sm.add_argument("--prior-versions", default=None)
    sm.add_argument("--loop-noise", type=float, default=None,
                    help="A/B LOOP MODE noise floor (default tools/ab_loop.py NOISE)")
    sm.add_argument("--loop-out", default=None, help="the A/B LOOP MODE verdicts as JSON (ab-loop-verdicts.json)")
    sm.add_argument("--out", default=None)
    return p


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("plan", "chunks", "prep-groups", "summary", "timed-check", "empty-board", "ab-gaps"):
        a = build_box_parser().parse_args(argv)
        return {"plan": cmd_plan, "chunks": cmd_chunks, "prep-groups": cmd_prep_groups, "summary": cmd_summary,
                "timed-check": cmd_timed_check, "empty-board": cmd_empty_board, "ab-gaps": cmd_ab_gaps}[a.cmd](a)
    return cmd_select(build_select_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
