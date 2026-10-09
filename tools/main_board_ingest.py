#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE MAIN BOARD: a rolling board per vendor column, the newest default race of
each lane x dataset on main, IN ADDITION to the release boards (Andrew,
2026-10-09: "a board that is just the latest as of everything even if not
coming from a pypi release").

    python3 tools/main_board_ingest.py --check      # dry run: the cells that would change
    python3 tools/main_board_ingest.py              # write bench/results/bench_board/main-<column>/

COLUMNS  nvidia-l40s (boxes nv, nv2) and amd-mi325x (box amd); Apple later.
LABEL    `main@<sha>` of the newest cell, never a wheel number. The board is
         unreleased: not reproducible by pip install; the release boards are
         the reference.

INPUTS (every path is an argument; the defaults are the grid collector's files)
  (a) lq results, `--results BOX=PATH` (default ~/mojolearn-evidence/grid-lq/
      {nv,nv2,amd}-results.txt): lines
      `<id> <vendor> main@<sha> ALGOS lane=<l> dataset=<d> arm=ours status=<s>
      median_ms=<ms> quality={...} digest=<h>` (the ALGOS_RE of
      tools/six_lane_grid_lq.py). A line with a `[...]` env bracket is an A/B
      or grid arm (MOJOLEARN_BUILD_DEFINES, MOJOLEARN_GRID_TAG, PREBUILT, ...)
      and is SKIPPED; so is any branch other than `--branch` (main).
  (b) board-runner summaries, `--grid-logs BOX=PATH` (default
      {nv,nv2,amd}-grid-logs.txt): the `GRIDBB ...` lines of
      tools/six_lane_grid_bb.py, prefixed by their log path
      (/root/lq/out/<id>/...). The job's CMD line in (a) decides: branch main,
      an env bracket holding ONLY MOJOLEARN_GRID_TAG (box_job.sh exports the
      CMD tag there), and a tag that is not a grid run's (`--skip-tag-re`:
      `g<8 hex>.` run ids, smoke tags). Any other MOJOLEARN_* token (build
      defines, an ENV= switch) means not the default configuration: skipped.
  (c) optional `--json-dir DIR`: bench_board board.json files found under DIR
      (an lq job's bb/board.json). The job id comes from the path
      (/<id>/bb/board.json) and must pass the (b) job rule; the commit
      (box.repo.commit) must be an ancestor of `--main-ref`. These cells keep
      their clock span (`upload_ms_separate`), so both of our clocks show.
  (d) the column's previous main board (its own cells re-enter the choice, so
      a rotated results file never drops a cell) and its LEDGER.json.

THE RULE: per (column, lane, dataset) the newest race wins, newest = highest
commit date of its sha (`git log -1 --format=%ct <sha>`; a sha not in this
repo is skipped with a note), ties by job number, over the runs whose status
is ok. A newer ok cell REPLACES an older one whatever the two times are: the
board never keeps an older number because it was lower. A run that is not ok
(error, refused, timeout, not_ready, NO-RECORD, NO-OURS-CELL) is never a
numeric cell and never replaces an ok cell (orchestrator, 2026-10-09): it goes
to the board's FAILED table (lane, dataset, sha, job, the result line's reason
text), and an older ok cell stays, flagged "newer run <sha>/<job> failed:
<reason>". Every replaced or failed observation goes to LEDGER.json /
LEDGER.md with its numbers and the cell that stayed. The Identity section at
the top of BOARD.md lists every DIFFER cell by name with both digests and the
sha.

EACH CELL: lane, dataset, mode identical, our median (one scored run), the
quality metrics, the output digest, commit sha, box/job id, commit date, and
the opponents COPIED from the stored opponent boards (`--opponents
COLUMN=LABEL=PATH`, in priority order; never re-run here), with the stored
cells' clock spans. The board is written THROUGH tools/bench_board.py
(add_ratios, save_result, write_board -> render_board, which draws both clocks
from tools/board_clock_audit.py: torch GPU arms kernel/kernel, every other arm
whole/whole, never an invented copy time). Identity: the digest of the same
lane x dataset x sha on the other vendor, MATCH / DIFFER / n/a, in the cell's
`main_board.identity` and in its status column.

Opponents are withheld (not copied) for an algos or classical2 race whose lane
settings at this tree's HEAD differ from the settings the opponent race
recorded: a ratio against different settings would not compare. The race
says so on the board.

This is tooling (standard library only); it times nothing and runs nothing on
a box. The board never states a direction.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import importlib.util
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
EVIDENCE = os.path.expanduser("~/mojolearn-evidence")
GRID_LQ = os.path.join(EVIDENCE, "grid-lq")
BOARD_ROOT = os.path.join(REPO, "bench", "results", "bench_board")

SCHEMA_NOTE = "mojolearn-main-board/1"
VENDOR_COLUMN = {"nvidia": "nvidia-l40s", "amd": "amd-mi325x"}
COLUMN_VENDOR = {v: k for k, v in VENDOR_COLUMN.items()}
COLUMN_GPU = {"nvidia-l40s": {"vendor": "nvidia", "api": "cuda", "name": "NVIDIA L40S"},
              "amd-mi325x": {"vendor": "amd", "api": "hip", "name": "AMD Instinct MI325X"}}
BOX_TEXT = {"nv": "nv (RunPod L40S)", "nv2": "nv2 (RunPod L40S)", "amd": "amd (DigitalOcean MI325X)"}
OTHER_VENDOR = {"nvidia": "amd", "amd": "nvidia"}

DEFAULT_RESULTS = ["%s=%s" % (b, os.path.join(GRID_LQ, "%s-results.txt" % b)) for b in ("nv", "nv2", "amd")]
DEFAULT_GRID_LOGS = ["%s=%s" % (b, os.path.join(GRID_LQ, "%s-grid-logs.txt" % b)) for b in ("nv", "nv2", "amd")]
#: The stored opponent boards, per column, highest priority first: the Oct 6
#: opponent re-score (carries the clock spans of AGENTS.md item 6), then the
#: release boards' own opponent cells (0.8.25, board-resume-r2).
DEFAULT_OPPONENTS = [
    "nvidia-l40s=opponents-default-20261006=" + os.path.join(BOARD_ROOT, "overnight-20261006-nvidia-default", "board.json"),
    "nvidia-l40s=opponents-specific-20261006=" + os.path.join(BOARD_ROOT, "overnight-20261006-nvidia-specific", "board.json"),
    "nvidia-l40s=release-board-resume-r2=" + os.path.join(EVIDENCE, "board-resume-r2", "nvidia-l40s", "board", "board.json"),
    "amd-mi325x=opponents-20261006=" + os.path.join(BOARD_ROOT, "overnight-20261006-amd", "board.json"),
    "amd-mi325x=release-board-resume-r2=" + os.path.join(EVIDENCE, "board-resume-r2", "amd-mi325x", "board", "board.json"),
]
#: Grid run tags (tools/six_lane_grid_lq.py run_id_of: 'g' + 8 hex, then '.<pack>') and smoke tags.
DEFAULT_SKIP_TAG_RE = r"^g[0-9a-f]{8}[A-Za-z0-9_-]*\.|smoke"

# The line grammar of tools/six_lane_grid_lq.py (RESULT_RE, ALGOS_RE, DIGEST_TAIL_RE, GRIDBB_RE), copied so
# this tool loads no grid module; tools/test_main_board_ingest.py holds them to the same fixtures.
RESULT_RE = re.compile(r"^(?P<id>[A-Za-z]\d+) (?P<vendor>nvidia|amd) (?P<branch>\S+?)@(?P<head>[0-9a-f]+)(?P<rest>.*)$")
ALGOS_RE = re.compile(r"(?:^|\s)ALGOS lane=(?P<lane>\S+) dataset=(?P<ds>\S+) arm=(?P<arm>\S+) status=(?P<status>\S+) "
                      r"median_ms=(?P<ms>\S+)(?: quality=(?P<q>.*))?$")
DIGEST_TAIL_RE = re.compile(r" digest=([0-9a-f]+|none)$")
GRIDBB_RE = re.compile(r"^GRIDBB tag=(?P<tag>\S+) vendor=(?P<vendor>\S+) head=(?P<head>\S+) family=(?P<family>\S+) "
                       r"lane=(?P<lane>\S+) dataset=(?P<ds>\S+) status=(?P<status>\S+) median_ms=(?P<ms>\S+) "
                       r"hash=(?P<hash>\S+) quality=(?P<q>.*)$")
CMD_RE = re.compile(r"^\s*CMD (?P<tag>\S+) rc=(?P<rc>-?\d+)")
JOB_IN_PATH_RE = re.compile(r"(?:^|/)([A-Za-z]\d{3,})/")
DIGEST_CHARS = 16          # box_job.sh keeps the first 16 hex characters of a digest
TAG_ENV = "MOJOLEARN_GRID_TAG"
INFRA = ("not_ready", "NO-RECORD", "NO-OURS-CELL")
#: Lane-settings keys that change what a race measures (bench_board_algos.lane_config,
#: bench_board_more.LANE_CONFIG); descriptive text keys are not compared.
SETTINGS_KEYS = ("params", "dataset_params", "block", "task", "kind", "stride_subsets", "sklearn", "cuml", "rows")
PRIORITY = {"json": 3, "algos": 2, "gridbb": 2, "previous": 1}
OB_FIELDS = ("column", "vendor", "box", "job", "sha", "ct", "kind", "family", "lane", "dataset", "race_id",
             "status_raw", "status", "infra", "median_ms", "quality", "digest", "comparability", "evidence",
             "measured", "tag", "reason")


def now_utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def iso(ct):
    if ct is None:
        return None
    return datetime.datetime.fromtimestamp(int(ct), datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_bench_board():
    spec = importlib.util.spec_from_file_location("bench_board_for_main_board", os.path.join(HERE, "bench_board.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# git: commit dates
# ---------------------------------------------------------------------------

class GitResolver:
    """sha prefix -> (full sha, commit time) or None when the repo lacks it."""

    def __init__(self, repo, main_ref="origin/main"):
        self.repo, self.main_ref = repo, main_ref
        self.cache, self.anc = {}, {}

    def __call__(self, sha):
        if sha not in self.cache:
            p = subprocess.run(["git", "-C", self.repo, "log", "-1", "--no-walk", "--format=%H %ct", sha, "--"],
                               capture_output=True, text=True)
            parts = p.stdout.split()
            self.cache[sha] = (parts[0], int(parts[1])) if p.returncode == 0 and len(parts) == 2 else None
        return self.cache[sha]

    def on_main(self, full):
        if full not in self.anc:
            p = subprocess.run(["git", "-C", self.repo, "merge-base", "--is-ancestor", full, self.main_ref],
                               capture_output=True, text=True)
            self.anc[full] = p.returncode == 0
        return self.anc[full]


# ---------------------------------------------------------------------------
# Parsing (pure: text in, records out)
# ---------------------------------------------------------------------------

def _num(text):
    try:
        v = float(text)
    except (TypeError, ValueError):
        return None
    return v if v == v and v not in (float("inf"), float("-inf")) else None


def split_bracket(rest):
    """' [ K=V K2=V2 ] tail' -> ({K: V, ...}, ' tail'); no bracket -> (None, rest).
    A bracket word without '=' is kept under '_words'."""
    s = rest.lstrip()
    if not s.startswith("["):
        return None, rest
    end = s.find("]")
    if end < 0:
        return {"_words": [s[1:]]}, ""
    env = {}
    for tok in s[1:end].split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            env[k] = v
        else:
            env.setdefault("_words", []).append(tok)
    return env, " " + s[end + 1:].lstrip()


def parse_algos(text):
    m = ALGOS_RE.search(text)
    if not m:
        return None
    q, digest = m.group("q"), None
    if q is not None:
        d = DIGEST_TAIL_RE.search(q)
        if d:
            digest = None if d.group(1) == "none" else d.group(1)[:DIGEST_CHARS]
            q = q[:d.start()]
        try:
            q = json.loads(q)
        except ValueError:
            q = None
    return dict(lane=m.group("lane"), dataset=m.group("ds"), arm=m.group("arm"), status=m.group("status"),
                median_ms=_num(m.group("ms")), quality=q if isinstance(q, dict) else {}, digest=digest)


def parse_results(lines, box, evidence):
    """results.txt lines of one box -> (jobs {id: job}, ALGOS records [..])."""
    jobs, algos = {}, []
    for raw in lines:
        m = RESULT_RE.match(raw.strip())
        if not m:
            continue
        jid = m.group("id")
        env, tail = split_bracket(m.group("rest"))
        job = jobs.setdefault(jid, dict(box=box, id=jid, vendor=m.group("vendor"), branch=m.group("branch"),
                                        head=m.group("head"), env={}, bracket=False, cmd_tag=None))
        if env is not None:
            job["bracket"] = True
            job["env"].update(env)
        c = CMD_RE.match(tail)
        if c:
            job["cmd_tag"] = c.group("tag")
            continue
        a = parse_algos(tail)
        if a:
            algos.append(dict(a, box=box, job=jid, vendor=m.group("vendor"), branch=m.group("branch"),
                              head=m.group("head"), env=env, evidence=evidence))
    return jobs, algos


def parse_gridbb(lines, box, evidence):
    """grid-logs lines ('<log path>:GRIDBB tag=...') of one box -> GRIDBB records [..]."""
    out = []
    for raw in lines:
        i = raw.find("GRIDBB tag=")
        if i < 0:
            continue
        m = GRIDBB_RE.match(raw[i:].strip())
        if not m:
            continue
        jm = JOB_IN_PATH_RE.findall(raw[:i])
        try:
            q = json.loads(m.group("q"))
        except ValueError:
            q = {}
        h = m.group("hash")
        out.append(dict(box=box, job=jm[-1] if jm else None, tag=m.group("tag"), vendor=m.group("vendor"),
                        head=m.group("head"), family=m.group("family"), lane=m.group("lane"),
                        dataset=m.group("ds"), status=m.group("status"), median_ms=_num(m.group("ms")),
                        quality=q if isinstance(q, dict) else {},
                        digest=None if h in ("none", "None", "") else h[:DIGEST_CHARS], evidence=evidence))
    return out


def board_status(raw, median_ms):
    """A driver status -> the board's status text."""
    raw = str(raw)
    if raw == "ok":
        return "ok" if median_ms is not None else "REFUSED(ok without a median)"
    if raw.startswith(("REFUSED", "QUALITY", "HOST-MEMORY", "MODE-MISMATCH", "PARTIAL", "UNKNOWN")):
        return raw          # bench_board's own status text (GRIDBB writes its spaces as '_')
    return "REFUSED(%s)" % raw


def is_infra(raw):
    return any(str(raw).startswith(s) or str(raw).startswith("REFUSED(%s" % s) for s in INFRA)


# ---------------------------------------------------------------------------
# Admission: default configuration on main only
# ---------------------------------------------------------------------------

class Skips:
    def __init__(self):
        self.counts, self.notes = {}, []

    def add(self, reason, note=None):
        self.counts[reason] = self.counts.get(reason, 0) + 1
        if note and len(self.notes) < 200:
            self.notes.append("%s: %s" % (reason, note))


def job_is_default(job, branch, skip_tag_re):
    """None when a CMD job ran the default configuration on `branch`, else the reason."""
    if job is None:
        return "no job line in results"
    if job["branch"] != branch:
        return "branch %s" % job["branch"]
    env = job.get("env") or {}
    extra = sorted(k for k in env if k != TAG_ENV)
    if extra:
        return "A/B or switched build (%s)" % ",".join(extra)
    tag = env.get(TAG_ENV) or job.get("cmd_tag") or ""
    if skip_tag_re.search(tag):
        return "grid or smoke tag"
    return None


def make_ob(BB, resolve, skips, *, kind, vendor, box, job, head, family, lane, dataset, status_raw, median_ms,
            quality, digest, evidence, comparability=None, measured=None, tag=None, full=None):
    column = VENDOR_COLUMN.get(vendor)
    if column is None:
        skips.add("vendor without a main-board column", vendor)
        return None
    if full is None:
        r = resolve(head)
        if r is None:
            skips.add("sha not in this repo", "%s %s/%s %s %s/%s" % (head, box, job, kind, lane, dataset))
            return None
        full, ct = r
    else:
        ct = (resolve(full) or (full, None))[1]
        if ct is None:
            skips.add("sha not in this repo", "%s %s/%s %s %s/%s" % (full, box, job, kind, lane, dataset))
            return None
    shape = "full" if family == "neural" else None
    return dict(column=column, vendor=vendor, box=box, job=job, sha=full, ct=ct, kind=kind, family=family,
                lane=lane, dataset=dataset, race_id=BB.race_id(family, lane, dataset, None, shape),
                status_raw=status_raw, status=board_status(status_raw, median_ms), infra=is_infra(status_raw),
                median_ms=median_ms, quality=quality or {}, digest=digest, comparability=comparability or {},
                evidence=evidence, measured=measured, tag=tag, reason=failure_reason(status_raw, median_ms, quality))


def failure_reason(status_raw, median_ms, quality):
    """The failure text a result line carries: its status, plus any error/reason text its quality holds.
    None for an ok run with a median."""
    if str(status_raw) == "ok" and median_ms is not None:
        return None
    parts = [str(status_raw) if str(status_raw) != "ok" else "ok without a median"]
    for k in ("error", "reason", "refused", "message"):
        v = (quality or {}).get(k) if isinstance(quality, dict) else None
        if isinstance(v, str) and v.strip():
            parts.append("%s: %s" % (k, " ".join(v.split())[:240]))
    return "; ".join(parts)


def admit_algos(BB, rec, branch, resolve, skips):
    if rec["branch"] != branch:
        skips.add("ALGOS: branch other than %s" % branch)
        return None
    if rec["env"] is not None:
        keys = sorted(k for k in rec["env"] if k != "_words")
        skips.add("ALGOS: A/B or grid arm (%s)" % (",".join(keys) or "bracket"))
        return None
    if rec["arm"] != "ours":
        skips.add("ALGOS: arm %s" % rec["arm"])
        return None
    if rec["lane"] not in BB.ALGOS_LANES:
        skips.add("ALGOS: lane not on the board", rec["lane"])
        return None
    return make_ob(BB, resolve, skips, kind="algos", vendor=rec["vendor"], box=rec["box"], job=rec["job"],
                   head=rec["head"], family="algos", lane=rec["lane"], dataset=rec["dataset"],
                   status_raw=rec["status"], median_ms=rec["median_ms"], quality=rec["quality"],
                   digest=rec["digest"], evidence=rec["evidence"])


def admit_gridbb(BB, rec, jobs, branch, skip_tag_re, resolve, skips):
    job = jobs.get((rec["box"], rec["job"])) if rec["job"] else None
    why = job_is_default(job, branch, skip_tag_re)
    if why:
        skips.add("GRIDBB: " + why.split(" (")[0], "%s/%s %s" % (rec["box"], rec["job"], rec["tag"]))
        return None
    if not (job["head"].startswith(rec["head"]) or rec["head"].startswith(job["head"])):
        skips.add("GRIDBB: head differs from its job", "%s/%s %s vs %s" % (rec["box"], rec["job"], rec["head"], job["head"]))
        return None
    if rec["family"] not in BB.FAMILIES:
        skips.add("GRIDBB: unknown family", rec["family"])
        return None
    return make_ob(BB, resolve, skips, kind="gridbb", vendor=rec["vendor"], box=rec["box"], job=rec["job"],
                   head=job["head"], family=rec["family"], lane=rec["lane"], dataset=rec["dataset"],
                   status_raw=rec["status"], median_ms=rec["median_ms"], quality=rec["quality"],
                   digest=rec["digest"], evidence=rec["evidence"], tag=rec["tag"])


def json_observations(BB, json_dir, jobs, branch, skip_tag_re, resolve, skips):
    """bench_board board.json files under json_dir -> observations of our IDENTICAL cell per race."""
    out = []
    if not json_dir:
        return out
    by_id = {}
    for (box, jid), job in jobs.items():
        by_id.setdefault(jid, []).append(job)
    for root, _dirs, files in os.walk(json_dir):
        if "board.json" not in files:
            continue
        path = os.path.join(root, "board.json")
        ids = JOB_IN_PATH_RE.findall(path.replace(os.sep, "/"))
        jid = ids[-1] if ids else None
        cands = by_id.get(jid) or []
        job = cands[0] if len(cands) == 1 else None
        why = job_is_default(job, branch, skip_tag_re) if len(cands) <= 1 else "job id on several boxes"
        if why:
            skips.add("JSON: " + why.split(" (")[0], path)
            continue
        try:
            board = BB.load_result(path) or {}
        except (OSError, ValueError) as exc:
            skips.add("JSON: unreadable", "%s (%s)" % (path, exc.__class__.__name__))
            continue
        if (board.get("config") or {}).get("smoke"):
            skips.add("JSON: smoke run", path)
            continue
        commit = ((board.get("box") or {}).get("repo") or {}).get("commit") or ""
        r = resolve(commit) if commit else None
        if r is None or not resolve.on_main(r[0]) or not r[0].startswith(job["head"]):
            skips.add("JSON: commit not on main or not the job's", "%s %s" % (path, commit[:9]))
            continue
        vendor = ((board.get("box") or {}).get("gpu") or {}).get("vendor") or job["vendor"]
        for rid, rr in sorted((board.get("races") or {}).items()):
            cell = next((c for c in rr.get("cells") or [] if c.get("arm") == "ours" and c.get("mode") == "identical"),
                        None)
            if cell is None or cell.get("rows_tag") not in (None, "full"):
                skips.add("JSON: race without a full-size IDENTICAL cell of ours", rid)
                continue
            ob = make_ob(BB, resolve, skips, kind="json", vendor=vendor, box=job["box"], job=jid, head=job["head"],
                         family=rr.get("family"), lane=rr.get("lane"), dataset=rr.get("dataset"),
                         status_raw=cell.get("status"), median_ms=cell.get("median_ms"),
                         quality=cell.get("quality"), digest=(cell.get("hash") or "")[:DIGEST_CHARS] or None,
                         evidence=path, comparability=cell.get("comparability"), measured=rr.get("finished"),
                         full=r[0])
            if ob is not None:
                ob["status"] = cell.get("status") or ob["status"]
                out.append(ob)
    return out


def previous_observations(prev):
    """The cells of ours on the column's previous main board, as observations."""
    out = []
    for rr in ((prev or {}).get("races") or {}).values():
        for c in rr.get("cells") or []:
            mb = c.get("main_board") or {}
            ob = mb.get("observation")
            if c.get("library") == "mojolearn" and isinstance(ob, dict) and ob.get("sha"):
                out.append(dict({k: ob.get(k) for k in OB_FIELDS}, prior_kind=ob.get("kind"), kind="previous"))
    # the failed runs the previous board listed (FAILED table, flags), so a rotated input keeps them too
    for ob in ((prev or {}).get("main_board") or {}).get("failed_observations") or []:
        if isinstance(ob, dict) and ob.get("sha"):
            out.append(dict({k: ob.get(k) for k in OB_FIELDS}, prior_kind=ob.get("kind"), kind="previous"))
    return out


# ---------------------------------------------------------------------------
# Choice: newest wins
# ---------------------------------------------------------------------------

def job_number(job):
    m = re.search(r"\d+", str(job or ""))
    return int(m.group(0)) if m else -1


def order_key(ob):
    return (ob.get("ct") or 0, job_number(ob.get("job")), str(ob.get("job") or ""), PRIORITY.get(ob["kind"], 0))


def dedupe(observations):
    """One observation per (vendor, box, job, race): a fresh parse over the previous board, a JSON cell
    over its summary line."""
    best = {}
    for ob in observations:
        k = (ob["vendor"], ob["box"], ob["job"], ob["race_id"])
        if k not in best or PRIORITY.get(ob["kind"], 0) >= PRIORITY.get(best[k]["kind"], 0):
            best[k] = ob
    return list(best.values())


def ident_group(ob):
    kind = source_kind(ob)
    return "algos-race" if kind == "algos" else "board"


def short(ob):
    return dict(sha=ob["sha"], commit_date=iso(ob.get("ct")), box=ob["box"], job=ob["job"], kind=source_kind(ob),
                status=ob["status"], median_ms=ob.get("median_ms"), digest=ob.get("digest"),
                evidence=ob.get("evidence"), reason=ob.get("reason"))


def source_kind(ob):
    """The input an observation came from (a re-read previous-board observation keeps its first kind)."""
    return (ob.get("prior_kind") or "previous") if ob["kind"] == "previous" else ob["kind"]


def stored_ob(ob):
    return dict({k: ob.get(k) for k in OB_FIELDS}, kind=source_kind(ob))


def is_ok(ob):
    """Only an ok run with a median becomes a numeric board cell (orchestrator, 2026-10-09): error,
    refused, timeout, not_ready, NO-RECORD and NO-OURS-CELL never do, and never replace an ok cell."""
    return ob.get("status") == "ok" and ob.get("median_ms") is not None


def choose(observations):
    """-> (winners {(column, race_id): newest ok ob}, failures {(column, race_id): [failed obs newer than
    the winner, or all of them when the race has no ok run], newest first}, ledger entries [..])."""
    groups = {}
    for ob in observations:
        groups.setdefault((ob["column"], ob["race_id"]), []).append(ob)
    winners, failures, ledger = {}, {}, []
    for key, group in sorted(groups.items()):
        group.sort(key=order_key, reverse=True)
        ok = [o for o in group if is_ok(o)]
        win = ok[0] if ok else None
        if win is not None:
            winners[key] = win
        newer_failed = [o for o in group if not is_ok(o) and (win is None or order_key(o) > order_key(win))]
        if newer_failed:
            failures[key] = newer_failed
        for o in group:
            if o is win:
                continue
            failed = not is_ok(o)
            if failed and o["infra"]:
                reason = "infrastructure status %s never replaces an ok cell" % o["status_raw"]
            elif failed:
                reason = "failed run (%s) never replaces an ok cell" % (o.get("reason") or o["status"])
            elif o.get("ct") == win.get("ct"):
                reason = "same commit date, earlier job"
            else:
                reason = "older commit"
            if failed and win is None:
                reason += "; no ok run of this race: FAILED table"
            elif failed and order_key(o) > order_key(win):
                reason += "; flagged on the ok cell from main@%s" % win["sha"][:9]
            ledger.append(dict(column=key[0], race=key[1], reason=reason, failed=failed, replaced=short(o),
                               replaced_by=short(win) if win else None))
    return winners, failures, ledger


def identity_index(observations):
    idx = {}
    for ob in observations:
        if is_ok(ob) and ob.get("digest"):
            idx.setdefault((ob["vendor"], ob["race_id"], ob["sha"], ident_group(ob)), []).append(ob)
    return idx


def identity(ob, idx):
    other = OTHER_VENDOR.get(ob["vendor"])
    cands = idx.get((other, ob["race_id"], ob["sha"], ident_group(ob))) or []
    if not cands or not ob.get("digest") or ob["status"] != "ok":
        return dict(status="n/a", other_column=VENDOR_COLUMN.get(other), other_digest=None, other_job=None)
    o = max(cands, key=order_key)
    same = o["digest"][:DIGEST_CHARS] == ob["digest"][:DIGEST_CHARS]
    return dict(status="MATCH" if same else "DIFFER", other_column=VENDOR_COLUMN.get(other),
                other_digest=o["digest"], other_job="%s/%s" % (o["box"], o["job"]))


# ---------------------------------------------------------------------------
# Opponents: copied from the stored opponent boards
# ---------------------------------------------------------------------------

def load_opponent_sources(BB, specs, column, skips):
    out = []
    for spec in specs:
        col, label, path = spec.split("=", 2)
        if col != column:
            continue
        if not os.path.exists(path):
            skips.add("opponent board missing", "%s %s" % (column, path))
            continue
        out.append(dict(label=label, path=path, board=BB.load_result(path) or {}))
    return out


def _settings_view(cfg):
    if not isinstance(cfg, dict):
        return None
    return json.dumps({k: cfg.get(k) for k in SETTINGS_KEYS if k in cfg}, sort_keys=True, default=str)


def current_lane_config(BB, family, lane):
    try:
        if family == "algos":
            return BB.ALGOS.lane_config(lane)
        if family == "classical2":
            return BB.MORE.LANE_CONFIG.get(lane)
    except (KeyError, AttributeError):
        return None
    return None


def copy_opponents(BB, sources, family, lane, race_id):
    """-> (opponent cells, the first source race (for lane_config), withheld reason or None)."""
    cells, first, seen = [], None, set()
    cur = _settings_view(current_lane_config(BB, family, lane))
    for src in sources:
        rr = ((src["board"].get("races") or {}).get(race_id))
        if not rr:
            continue
        if cur is not None and _settings_view(rr.get("lane_config")) not in (None, cur):
            return [], rr, ("opponents withheld: the lane settings at this tree's HEAD differ from the settings "
                            "%s recorded for its opponent race (an opponent job must score them again)" % src["label"])
        box = src["board"].get("box") or {}
        for c in rr.get("cells") or []:
            if c.get("library") == "mojolearn" or str(c.get("arm", "")).startswith("ours") or c.get("arm") in seen:
                continue
            seen.add(c.get("arm"))
            oc = copy.deepcopy(c)
            for k in ("ratio_ours_identical_over", "ratio_ours_fast_over", "ratio_ours_identical_clock",
                      "ratio_ours_fast_clock", "clock"):
                oc.pop(k, None)
            oc["copied_from"] = {"board": src["label"], "path": src["path"], "race": race_id,
                                 "box": (box.get("host") or {}).get("hostname"),
                                 "gpu": (box.get("gpu") or {}).get("name"),
                                 "measured": (c.get("stored") or {}).get("measured_at") or rr.get("finished")}
            old = oc.get("source")
            oc["source"] = "copied from %s%s" % (src["label"], ("; " + old) if old else "")
            cells.append(oc)
        if first is None:
            first = rr
    return cells, first, None


# ---------------------------------------------------------------------------
# Assembly: the result structure tools/bench_board.py writes and renders
# ---------------------------------------------------------------------------

def failed_flag(o):
    return "newer run %s/%s failed: %s" % (o["sha"][:9], o["job"], o.get("reason") or o["status"])


def our_cell(BB, ob, ident, newer_failed=()):
    ms = ob.get("median_ms") if ob["status"] == "ok" else None
    tail = "main@%s %s/%s %s; identity vs %s: %s" % (
        ob["sha"][:9], ob["box"], ob["job"], (iso(ob.get("ct")) or "?")[:10], ident["other_column"], ident["status"])
    flags = [failed_flag(o) for o in newer_failed]
    if flags:
        tail += "; " + "; ".join(flags)
    return {
        "family": ob["family"], "lane": ob["lane"], "dataset": ob["dataset"], "rows": None, "rows_tag": "full",
        "neural_shape": "full" if ob["family"] == "neural" else None,
        "arm": "ours", "library": "mojolearn", "mode": "identical",
        "device": BB.arm_device("ours", ob["vendor"]),
        "settings": {"rounds": 1, "source": "main board: one scored run of ours (%s)" % ob["kind"]},
        "times_ms": [ms] if ms is not None else [], "warmup_ms": None,
        "median_ms": ms, "min_ms": ms, "max_ms": ms, "rounds": 1 if ms is not None else 0,
        "status": ob["status"], "quality": ob.get("quality") or {}, "hash": ob.get("digest"), "hash_stable": None,
        "verdict": "main board, one scored run", "comparability": ob.get("comparability") or {},
        "peak_host_mb": None, "peak_gpu_mb": None, "memory": {},
        "source": tail,
        "main_board": {"sha": ob["sha"], "commit_date": iso(ob.get("ct")), "box": ob["box"], "job": ob["job"],
                       "measured": ob.get("measured"), "identity": ident, "newer_failed": flags,
                       "observation": stored_ob(ob)},
    }


def _md(v):
    return " ".join(str(v if v is not None else "-").split()).replace("|", "/")


def identity_section(races):
    """The identity table: every DIFFER cell by name first (a cross-vendor difference is the board's most
    important fact), then the counts."""
    differ, counts = [], {}
    for rid in sorted(races):
        c = next(x for x in races[rid]["cells"] if x.get("library") == "mojolearn")
        ident = c["main_board"]["identity"]
        counts[ident["status"]] = counts.get(ident["status"], 0) + 1
        if ident["status"] == "DIFFER":
            differ.append((rid, c, ident))
    L = ["Same lane, dataset and commit on the other GPU vendor (identity = equal output digests on NVIDIA "
         "and AMD). Counts: %s." % (", ".join("%s %d" % kv for kv in sorted(counts.items())) or "none"), ""]
    if differ:
        L += ["**DIFFER: %d cells whose digest differs from the other vendor's at the same commit.**" % len(differ),
              "", "| race | commit | this column: box/job, digest | other column: box/job, digest |",
              "|---|---|---|---|"]
        for rid, c, ident in differ:
            mb = c["main_board"]
            L.append("| %s | main@%s | %s/%s %s | %s %s %s |" % (
                _md(rid), _md(mb["sha"][:9]), _md(mb["box"]), _md(mb["job"]), _md(c.get("hash")),
                _md(ident["other_column"]), _md(ident["other_job"]), _md(ident["other_digest"])))
    else:
        L.append("DIFFER: none.")
    return {"title": "Identity", "lines": L}


def failed_section(failures, winners, column):
    rows = []
    for (col, rid), obs in sorted(failures.items()):
        if col != column:
            continue
        win = winners.get((col, rid))
        for o in obs:
            rows.append("| %s | %s | main@%s | %s/%s | %s | %s |" % (
                _md(o["lane"]), _md(o["dataset"]), _md(o["sha"][:9]), _md(o["box"]), _md(o["job"]),
                _md(o.get("reason") or o["status"]),
                "main@%s %s/%s stays" % (win["sha"][:9], win["box"], win["job"]) if win else "none"))
    L = ["Runs on main whose status is not ok (error, refused, timeout, not_ready, NO-RECORD, NO-OURS-CELL). "
         "They are never a numeric cell and never replace an ok cell; an older ok cell stays on the board "
         "flagged with the failed run. %d failed runs." % len(rows), ""]
    if rows:
        L += ["| lane | dataset | commit | box/job | reason | ok cell on the board |",
              "|---|---|---|---|---|---|"] + rows
    return {"title": "FAILED", "lines": L}, len(rows)


def assemble(BB, column, winners, ledger, idx, sources, inputs, generated, failures=None):
    failures = failures or {}
    races, withheld, boxes, shas = {}, 0, set(), []
    for (col, rid), ob in sorted(winners.items()):
        if col != column:
            continue
        ident = identity(ob, idx)
        cell = our_cell(BB, ob, ident, failures.get((col, rid)) or ())
        opps, src_rr, why = copy_opponents(BB, sources, ob["family"], ob["lane"], rid)
        # our cell without a quality (the host reference refused) whose output hash equals a copied opponent's:
        # quality {"identical_to": <arm>} (tools/bench_board.py stored_identity_quality; a digest compare only)
        cells = BB.add_ratios(BB.stored_identity_quality([cell] + opps))
        lane_config = copy.deepcopy((src_rr or {}).get("lane_config")) if src_rr else None
        if why:
            withheld += 1
            lane_config = copy.deepcopy(current_lane_config(BB, ob["family"], ob["lane"])) or {}
            lane_config["mismatches"] = list(lane_config.get("mismatches") or []) + [why]
        rr = {"id": rid, "family": ob["family"], "lane": ob["lane"], "dataset": ob["dataset"], "rows": None,
              "shape": "full" if ob["family"] == "neural" else None,
              "status": "done" if ob["status"] == "ok" else "failed", "rc": None,
              "log": "%s (%s/%s)" % (ob.get("evidence"), ob["box"], ob["job"]),
              "host": {"hostname": BOX_TEXT.get(ob["box"], ob["box"]) + " job " + str(ob["job"])},
              "cells": cells, "infer_cells": [],
              "params_check": "not checked on the main board (our one scored run; opponents copied, never re-run)",
              "opponent_source": {"withheld": why} if why else
              {"boards": sorted({c["copied_from"]["board"] for c in opps})},
              "main_board": {"sha": ob["sha"], "box": ob["box"], "job": ob["job"], "identity": ident["status"]}}
        if lane_config:
            rr["lane_config"] = lane_config
        # neural lanes: the headline is ours IDENTICAL over torch's fastest bf16 arm, the fp32 twin beside it
        # (tools/bench_board.py neural_headline; render_board draws the table and each race's headline line)
        hl = BB.neural_headline(rr)
        if hl is not None:
            rr["neural_headline"] = hl
        races[rid] = rr
        boxes.add(ob["box"])
        shas.append((ob.get("ct") or 0, ob["sha"]))
    shas.sort()
    newest = shas[-1][1] if shas else None
    oldest = shas[0][1] if shas else None
    label = "main@%s" % newest[:9] if newest else "main@none"
    n_replaced = sum(1 for e in ledger if e["column"] == column)
    idn = {}
    for rr in races.values():
        s = rr["main_board"]["identity"]
        idn[s] = idn.get(s, 0) + 1
    notes = [
        "MAIN BOARD %s, version label %s. Unreleased: not reproducible by pip install; the release boards are "
        "the reference." % (column, label),
        "Cells: %d races; oldest cell main@%s (%s), newest cell main@%s (%s). Boxes: %s." % (
            len(races), (oldest or "none")[:9], iso(shas[0][0]) if shas else "-", (newest or "none")[:9],
            iso(shas[-1][0]) if shas else "-", ", ".join(BOX_TEXT.get(b, b) for b in sorted(boxes)) or "none"),
        "Rule: each lane x dataset shows the newest default-configuration race on main (highest commit date, "
        "then job number) whose status is ok. A newer ok cell replaces an older one whatever the two times "
        "are; a run that is not ok is never a numeric cell and never replaces an ok cell (FAILED table; an "
        "older ok cell stays, flagged with the newer failed run). %d replaced or failed observations are in "
        "LEDGER.md. A/B and grid arms (MOJOLEARN_BUILD_DEFINES, MOJOLEARN_GRID_TAG grid runs) are never on "
        "this board." % n_replaced,
        "Ours: one scored run per cell (lq RACE ALGOS lines, lq CMD bench_board summaries); the status column "
        "names the cell's commit, box/job, commit date and the other vendor's digest at the same commit "
        "(identity: %s)." % (", ".join("%s %d" % kv for kv in sorted(idn.items())) or "none"),
        "Opponents: copied from the stored opponent boards (%s), never re-run here; `ours IDENTICAL / arm` "
        "divides the two stored medians, and the clock columns read a torch GPU arm kernel/kernel and every "
        "other arm whole/whole (AGENTS.md measurement item 6). Our kernel clock is `-` unless the cell recorded "
        "upload_ms_separate. Opponents withheld for changed lane settings: %d races." % (
            ", ".join(s["label"] for s in sources) or "none found", withheld),
        "Neural lanes: the headline (its own table, and a line under each neural race) is ours IDENTICAL over "
        "torch's fastest bf16 arm, eager or compile, what customers run; the fp32 twin is the second column. "
        "Note: " + BB.IDENTITY_TAX_NOTE + ". A cell of ours whose output hash equals a copied opponent's shows "
        "quality identical_to=<arm> (the same bits) where the own-host reference gave none.",
    ]
    fsec, n_failed = failed_section(failures, winners, column)
    isec = identity_section(races)
    differ = [{"race": rid, "sha": rr["main_board"]["sha"],
               "digest": next(c for c in rr["cells"] if c.get("library") == "mojolearn").get("hash"),
               "job": "%s/%s" % (rr["main_board"]["box"], rr["main_board"]["job"]),
               "other": next(c for c in rr["cells"] if c.get("library") == "mojolearn")["main_board"]["identity"]}
              for rid, rr in sorted(races.items()) if rr["main_board"]["identity"] == "DIFFER"]
    gpu = dict(COLUMN_GPU[column])
    result = {
        "schema": BB.SCHEMA, "main_board_schema": SCHEMA_NOTE, "created": generated,
        "box": {"gpu": gpu, "host": {"hostname": ", ".join(BOX_TEXT.get(b, b) for b in sorted(boxes)) or None},
                "mojolearn": {"version": label,
                              "wheel": {"file": "none (unreleased; built from source at each cell's commit)",
                                        "sha256": "-"}},
                "repo": {"commit": newest}},
        "config": {"vendor": COLUMN_VENDOR[column], "modes": ["identical"], "rounds": 1, "smoke": False,
                   "families": sorted({rr["family"] for rr in races.values()})},
        "plan": sorted(races), "races": races, "board_notes": notes, "board_sections": [isec, fsec],
        "main_board": {"column": column, "label": label, "newest_sha": newest, "oldest_sha": oldest,
                       "boxes": sorted(boxes), "replaced": n_replaced, "withheld_opponents": withheld,
                       "identity": idn, "differ": differ, "failed_runs": n_failed, "inputs": inputs,
                       "failed_observations": [stored_ob(o) for (col, _rid), obs in sorted(failures.items())
                                               if col == column for o in obs],
                       "opponent_boards": [{"label": s["label"], "path": s["path"]} for s in sources]},
    }
    return result


def diff_boards(prev, new):
    """-> [(kind, race_id, old text, new text)] for cells of ours that would change."""
    def view(board):
        out = {}
        for rid, rr in ((board or {}).get("races") or {}).items():
            c = next((c for c in rr.get("cells") or [] if c.get("library") == "mojolearn"), None)
            if c is not None:
                mb = c.get("main_board") or {}
                out[rid] = (str(mb.get("sha") or "")[:9], "%s/%s" % (mb.get("box"), mb.get("job")),
                            c.get("median_ms"), c.get("status"), c.get("hash"),
                            "; ".join(mb.get("newer_failed") or []))
        return out
    a, b = view(prev), view(new)
    txt = lambda v: "main@%s %s %s ms %s digest %s%s" % (   # noqa: E731
        v[0], v[1], v[2], v[3], v[4], (" [%s]" % v[5]) if v[5] else "")
    out = []
    for rid in sorted(set(a) | set(b)):
        if rid not in a:
            out.append(("ADD", rid, "-", txt(b[rid])))
        elif rid not in b:
            out.append(("DROP", rid, txt(a[rid]), "-"))
        elif a[rid] != b[rid]:
            out.append(("CHANGE", rid, txt(a[rid]), txt(b[rid])))
    return out


def merge_ledger(old, new):
    seen, out = set(), []
    for e in list(new) + list(old or []):
        r = e.get("replaced") or {}
        k = (e.get("column"), e.get("race"), r.get("sha"), r.get("box"), r.get("job"), r.get("kind"))
        if k in seen:
            continue
        seen.add(k)
        out.append(e)
    return out


def render_ledger(column, entries):
    L = ["# Main board ledger: %s" % column, "",
         "Every observation the newest-wins rule did not put on the board, with its numbers and the cell that "
         "replaced it (tools/main_board_ingest.py). Times are milliseconds of one scored run.", "",
         "| race | reason | replaced: commit, box/job, ms, status, digest | by: commit, box/job, ms, status, digest |",
         "|---|---|---|---|"]
    fmt = lambda s: "-" if not s else "main@%s %s/%s %s %s %s" % (   # noqa: E731
        str(s.get("sha"))[:9], s.get("box"), s.get("job"), s.get("median_ms"), s.get("status"), s.get("digest"))
    for e in sorted(entries, key=lambda e: (e["race"], str((e.get("replaced") or {}).get("commit_date")))):
        L.append("| %s | %s | %s | %s |" % (e["race"], e["reason"], fmt(e.get("replaced")),
                                            fmt(e.get("replaced_by"))))
    return "\n".join(L).replace("faster", "[direction word removed]").replace("slower", "[direction word removed]") + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _pairs(specs):
    out = []
    for s in specs:
        box, path = s.split("=", 1)
        out.append((box, os.path.expanduser(path)))
    return out


def _read_lines(path, skips):
    try:
        with open(path, errors="replace") as fh:
            return fh.read().splitlines()
    except OSError:
        skips.add("input file missing", path)
        return []


def collect(BB, args, resolve, skips):
    jobs, recs, obs = {}, [], []
    skip_tag_re = re.compile(args.skip_tag_re)
    for box, path in _pairs(args.results):
        j, a = parse_results(_read_lines(path, skips), box, path)
        for jid, job in j.items():
            jobs[(box, jid)] = job
        recs.extend(a)
    for rec in recs:
        ob = admit_algos(BB, rec, args.branch, resolve, skips)
        if ob is not None:
            obs.append(ob)
    for box, path in _pairs(args.grid_logs):
        for rec in parse_gridbb(_read_lines(path, skips), box, path):
            ob = admit_gridbb(BB, rec, jobs, args.branch, skip_tag_re, resolve, skips)
            if ob is not None:
                obs.append(ob)
    obs.extend(json_observations(BB, args.json_dir, jobs, args.branch, skip_tag_re, resolve, skips))
    return obs


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", action="append", help="BOX=PATH lq results file (repeatable; default %s)"
                   % ", ".join(DEFAULT_RESULTS))
    p.add_argument("--grid-logs", action="append", help="BOX=PATH grid-logs file with GRIDBB lines (repeatable)")
    p.add_argument("--json-dir", help="directory of bench_board board.json files from lq jobs (optional)")
    p.add_argument("--opponents", action="append", help="COLUMN=LABEL=PATH stored opponent board, priority order")
    p.add_argument("--columns", default=",".join(VENDOR_COLUMN.values()))
    p.add_argument("--out-root", default=BOARD_ROOT, help="board roots go to <out-root>/main-<column>/")
    p.add_argument("--repo", default=REPO, help="git repo resolving commit dates")
    p.add_argument("--main-ref", default="origin/main")
    p.add_argument("--branch", default="main", help="the results' branch label admitted")
    p.add_argument("--skip-tag-re", default=DEFAULT_SKIP_TAG_RE)
    p.add_argument("--check", action="store_true", help="dry run: print the cells that would change, write nothing")
    p.add_argument("--max-lines", type=int, default=40, help="change lines printed per column with --check")
    return p


def run(argv=None, BB=None, resolve=None, out=sys.stdout):
    args = build_parser().parse_args(argv)
    args.results = args.results or DEFAULT_RESULTS
    args.grid_logs = args.grid_logs or DEFAULT_GRID_LOGS
    args.opponents = args.opponents or DEFAULT_OPPONENTS
    BB = BB or load_bench_board()
    resolve = resolve or GitResolver(args.repo, args.main_ref)
    skips = Skips()
    fresh = collect(BB, args, resolve, skips)
    generated = now_utc()
    inputs = {"results": args.results, "grid_logs": args.grid_logs, "json_dir": args.json_dir,
              "branch": args.branch, "generated": generated}
    columns = [c for c in args.columns.split(",") if c]
    prevs, all_obs = {}, list(fresh)
    for column in columns:
        if column not in COLUMN_VENDOR:
            raise SystemExit("main_board_ingest: unknown column %r (known: %s)" % (column, ", ".join(COLUMN_VENDOR)))
        root = os.path.join(args.out_root, "main-%s" % column)
        prevs[column] = (root, BB.load_result(os.path.join(root, "board.json")))
        all_obs.extend(previous_observations(prevs[column][1]))
    obs = dedupe(all_obs)
    winners, failures, ledger = choose(obs)
    idx = identity_index(obs)
    summary = {}
    for column in columns:
        root, prev = prevs[column]
        sources = load_opponent_sources(BB, args.opponents, column, skips)
        result = assemble(BB, column, winners, ledger, idx, sources, inputs, generated, failures)
        changes = diff_boards(prev, result)
        mine = [e for e in ledger if e["column"] == column]
        kinds = {}
        for k, _r, _a, _b in changes:
            kinds[k] = kinds.get(k, 0) + 1
        mb = result["main_board"]
        line = ("MAINBOARD column=%s label=%s races=%d add=%d change=%d drop=%d failed=%d ledger=%d "
                "withheld_opponents=%d identity=%s" % (
                    column, mb["label"], len(result["races"]), kinds.get("ADD", 0), kinds.get("CHANGE", 0),
                    kinds.get("DROP", 0), mb["failed_runs"], len(mine), mb["withheld_opponents"],
                    ",".join("%s:%d" % kv for kv in sorted(mb["identity"].items())) or "none"))
        print(line, file=out)
        for d in mb["differ"]:
            print("  DIFFER %s main@%s %s %s vs %s %s %s" % (
                d["race"], d["sha"][:9], d["job"], d["digest"], d["other"]["other_column"],
                d["other"]["other_job"], d["other"]["other_digest"]), file=out)
        summary[column] = line
        if args.check:
            for k, rid, a, b in changes[:args.max_lines]:
                print("  %s %s: %s -> %s" % (k, rid, a, b), file=out)
            if len(changes) > args.max_lines:
                print("  ... %d more" % (len(changes) - args.max_lines), file=out)
            continue
        os.makedirs(root, exist_ok=True)
        BB.save_result(os.path.join(root, "board.json"), result)
        BB.write_board(root, result)
        old_ledger = []
        lp = os.path.join(root, "LEDGER.json")
        if os.path.exists(lp):
            with open(lp) as fh:
                old_ledger = (json.load(fh) or {}).get("entries") or []
        entries = merge_ledger(old_ledger, mine)
        with open(lp + ".tmp", "w") as fh:
            json.dump({"schema": SCHEMA_NOTE, "column": column, "updated": generated, "entries": entries},
                      fh, indent=1, sort_keys=True, default=str)
        os.replace(lp + ".tmp", lp)
        with open(os.path.join(root, "LEDGER.md"), "w") as fh:
            fh.write(render_ledger(column, entries))
        with open(os.path.join(root, "INGEST.json"), "w") as fh:
            json.dump({"generated": generated, "inputs": inputs, "summary": line, "skipped": skips.counts,
                       "skip_notes": skips.notes}, fh, indent=1, sort_keys=True)
    print("MAINBOARD-SKIPS %s" % json.dumps(skips.counts, sort_keys=True), file=out)
    for n in skips.notes[:10]:
        print("  skip %s" % n, file=out)
    if len(skips.notes) > 10:
        print("  ... %d more skip notes%s" % (len(skips.notes) - 10, "" if args.check else " (INGEST.json)"), file=out)
    return summary


def main(argv=None):
    run(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
