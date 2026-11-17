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
    summary  --compare C --rows R --results RES --ids F [--units U]: SUMMARY-TG (rows timed on both,
             FASTER/SAME/REGRESSION, rows still over 1.10 worst first, rows fixed (over 1.10 in the input compare, now
             at or under 1.00), done/total per arm, and each owed A/B unit's arm over its back-to-back base).

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
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "ops"))
import paper_compare as PC  # noqa: E402
import race_fairness as RF  # noqa: E402

REASONS = ("paper", "slower", "changed", "changed?", "internal")
#: the job's fixed cost outside the timed races, seconds (d95f805cce full board, L40S, job/phases.tsv: probe 2, plan
#: 0, install 8-46, compare 1): rounded up
FIXED_S = 90
#: untimed smoke seconds per race and arm: the d95f805cce all-smoke phase ran 650 candidate + 627 0.8.55 races in
#: 805 s of parallel shards
SMOKE_S_PER_RACE = 0.65
#: the start of one timed bench_board process (interpreter, plan, preflight audit, smoke gate) per chunk and arm:
#: NOT MEASURED, an assumption the first targeted job confirms (job.log shows each chunk's process time)
PROC_S = 30
DEFAULT_CAP_MIN = 230       # under tools/nvidia_central.sh's 240-minute cap, with the job's own teardown
DEFAULT_CHUNK_S = 600
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
def estimate(rows, chunk_s=DEFAULT_CHUNK_S):
    """{candidate_s, prior_s, smoke_s, overhead_s, chunks, total_s, total_min} of a job over ROWS."""
    new = sum(r.get("new_s") or 0.0 for r in rows)
    prior = sum(r.get("prior_s") or 0.0 for r in rows)
    nchunks = len(chunk_ids([r["id"] for r in rows], {r["id"]: r for r in rows}, chunk_s))
    smoke = SMOKE_S_PER_RACE * (len(rows) + sum(1 for r in rows if (r.get("prior_s") or 0) > 0))
    over = FIXED_S + 2 * PROC_S * nchunks
    total = new + prior + smoke + over
    return {"candidate_s": new, "prior_s": prior, "smoke_s": smoke, "overhead_s": over, "chunks": nchunks,
            "total_s": total, "total_min": total / 60.0}


def estimate_line(est):
    return ("TG-ESTIMATE minutes total=%.0f candidate=%.0f 0.8.55=%.0f smoke=%.0f overhead=%.0f (chunks=%d, %d s per "
            "process assumed)" % (est["total_min"], est["candidate_s"] / 60, est["prior_s"] / 60, est["smoke_s"] / 60,
                                  est["overhead_s"] / 60, est["chunks"], PROC_S))


def split_rows(rows, cap_min, chunk_s=DEFAULT_CHUNK_S):
    """ROWS in file order cut into parts whose estimate is under CAP_MIN each (a row alone over the cap is a part of
    its own: it cannot be split further)."""
    parts, cur = [], []
    for r in rows:
        if cur and estimate(cur + [r], chunk_s)["total_min"] > cap_min:
            parts.append(cur)
            cur = []
        cur.append(r)
    if cur:
        parts.append(cur)
    return parts


def check_cap(rows_path, cap_min, chunk_s=DEFAULT_CHUNK_S, write_split=True):
    """(estimate, [split file paths]) of a rows file; the split list is empty when it fits under CAP_MIN. The parts
    are written beside it as <stem>.part<i>.tsv, each one complete job (both arms on the same box)."""
    rows = read_rows(rows_path)
    est = estimate(rows, chunk_s)
    if est["total_min"] <= cap_min:
        return est, []
    p = Path(rows_path)
    files = []
    for i, part in enumerate(split_rows(rows, cap_min, chunk_s), 1):
        f = p.with_name("%s.part%d%s" % (p.stem, i, p.suffix or ".tsv"))
        if write_split:
            PC.write_atomic(str(f), rows_text(part, ["part %d of %s (cap %s min): %s" % (
                i, p.name, cap_min, estimate_line(estimate(part, chunk_s)))]))
        files.append((str(f), estimate(part, chunk_s)))
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
    (kind: recipe / extra / internal). A recipe row's ratio counts only with a verdict flag (an on-box baseline)."""
    pv = doc.get("paper_version") or PC.PAPER_VERSION
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


def cmd_select(a):
    planner = PC.load_planner(a.planner)
    plan_ids = PC.plan_all(planner, a.vendor)[0]
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
    print(estimate_line(estimate([by_id[i] for i in ids], a.chunk_s)))
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


def _done(board_path, ids):
    try:
        races = (PC.load_json(board_path).get("races") or {})
    except (OSError, ValueError):
        races = {}
    return sum(1 for i in ids if (races.get(i) or {}).get("status") == "done")


def summary(doc, rows, results, ids, units=(), prior_versions=None):
    """The SUMMARY-TG text (module docstring)."""
    import measure_owed as MO
    want = {r["id"] for r in rows}
    input_ratio = {r["id"]: r.get("input_ratio") for r in rows}
    pv = doc.get("paper_version") or PC.PAPER_VERSION
    lines = ["TARGETED vendor=%s %s -> %s rows=%d (%s)" % (doc.get("vendor"), pv, doc.get("new_label"), len(rows),
                                                          doc.get("verdict_basis") or "no verdict")]
    counts, over, now = {}, [], {}
    for rid, ratio, r, _kind in compare_rows(doc):
        if rid not in want:
            continue
        flags = [f for f in r.get("flags") or [] if f in ("FASTER", "SAME", "REGRESSION")]
        if ratio is None or not flags:
            continue
        counts[flags[0]] = counts.get(flags[0], 0) + 1
        now[rid] = max(now.get(rid, ratio), ratio)
        if ratio > PC.SLOWER:
            over.append((ratio, r.get("key") or rid, r.get("new_ms"),
                         r.get("old_ms") if "old_ms" in r else ((r.get("priors") or {}).get(pv) or {}).get("ms")))
    lines.append("timed both=%d FASTER=%d SAME=%d REGRESSION=%d (rows of %d races)" % (
        sum(counts.values()), counts.get("FASTER", 0), counts.get("SAME", 0), counts.get("REGRESSION", 0), len(now)))
    lines.append("still over %.2f (%d), worst first:" % (PC.SLOWER, len(over)))
    for ratio, key, new, old in sorted(over, key=lambda x: -x[0]):
        lines.append("  %.2fx %s (%s -> %s ms)" % (ratio, key, PC.fmt_ms(old), PC.fmt_ms(new)))
    fixed = sorted(rid for rid, r in input_ratio.items() if r is not None and r > PC.SLOWER
                   and now.get(rid) is not None and now[rid] <= 1.0)
    lines.append("fixed (over %.2f in the input compare, now at or under 1.00): %d" % (PC.SLOWER, len(fixed)))
    lines += ["  %s (%.2fx -> %.2fx)" % (rid, input_ratio[rid], now[rid]) for rid in fixed]
    lines.append("done/total candidate %d/%d" % (_done(os.path.join(results, "board-tg", "board.json"), ids), len(ids)))
    for v in prior_versions or [pv]:
        pid = os.path.join(results, "plan-tg", "prior-%s" % v, "race-ids-pass.txt")
        lines.append("done/total %s %d/%d (of the listed races; %s passed its own smoke)" % (
            v, _done(os.path.join(results, "board-tg-prior-%s" % v, "board.json"), ids), len(ids),
            sum(1 for _ in open(pid)) if os.path.isfile(pid) else "?"))
    for arm, rows_tag, uids in units:
        base = MO.load_board(os.path.join(results, "board-tg-ab-%s-base-%s" % (arm, rows_tag), "board.json"))
        brd = MO.load_board(os.path.join(results, "board-tg-ab-%s-%s" % (arm, rows_tag), "board.json"))
        for rid in uids:
            bc = MO.our_cells(((base or {}).get("races") or {}).get(rid) or {})
            ac = MO.our_cells(((brd or {}).get("races") or {}).get(rid) or {})
            for k in sorted(set(bc) | set(ac), key=str):
                b, x = MO.cell_ms(bc.get(k)), MO.cell_ms(ac.get(k))
                r = x / b if b and x else None
                lines.append("AB %s rows=%s %s %s %s: base %s -> arm %s ms ratio %s%s" % (
                    arm, rows_tag, rid, k[0], k[1], PC.fmt_ms(b), PC.fmt_ms(x), PC.fmt_ratio(r),
                    "" if r is None else " " + MO.verdict(r, "FASTER", "SLOWER", "SAME")))
    return "\n".join(lines) + "\n"


def read_units(path):
    """[(B arm, rows tag, [ids])] of a tools/measure_owed.py units.tsv (the candidate units are the bases)."""
    out = []
    if not path or not os.path.isfile(path):
        return out
    with open(path) as fh:
        for ln in fh:
            f = ln.rstrip("\n").split("\t")
            if len(f) == 3 and f[0] not in ("candidate", "prior") and os.path.isfile(f[2]):
                with open(f[2]) as ih:
                    out.append((f[0], f[1], [x.strip() for x in ih if x.strip()]))
    return out


def cmd_summary(a):
    doc = PC.load_json(a.compare)
    with open(a.ids) as fh:
        ids = [x.strip() for x in fh if x.strip()]
    text = summary(doc, read_rows(a.rows), a.results, ids, read_units(a.units),
                   [v for v in (a.prior_versions or "").split(",") if v] or None)
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
    ch = sub.add_parser("chunks")
    ch.add_argument("--ids", required=True)
    ch.add_argument("--rows", default=None)
    ch.add_argument("--chunk-s", type=int, default=DEFAULT_CHUNK_S)
    ch.add_argument("--out-dir", required=True)
    sm = sub.add_parser("summary")
    sm.add_argument("--compare", required=True)
    sm.add_argument("--rows", required=True)
    sm.add_argument("--results", required=True)
    sm.add_argument("--ids", required=True, help="the listed races the board plans (plan-tg/race-ids-full.txt)")
    sm.add_argument("--units", default=None, help="the owed plan's units.tsv (the A/B arms)")
    sm.add_argument("--prior-versions", default=None)
    sm.add_argument("--out", default=None)
    return p


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("plan", "chunks", "summary"):
        a = build_box_parser().parse_args(argv)
        return {"plan": cmd_plan, "chunks": cmd_chunks, "summary": cmd_summary}[a.cmd](a)
    return cmd_select(build_select_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
