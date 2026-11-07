#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""WHERE THE TIME GOES: one read-only report over the timings every tool already writes (Andrew, 2026-11-07:
"do we measure everything in time? ... so we know where we need to make improvements").

    python3 tools/timing_ledger.py report [--kind test|group|step|lane|job|workflow] [--since 24h] [--top 50]
                                          [--by name|file|kind] [--json]
                                          [--tests DIR]... [--builds DIR]... [--release DIR|state.json]...
                                          [--identity DIR|record.json]... [--jobs DIR]... [--workflow run.json]...

It writes nothing and runs nothing; it reads, per kind of unit:

    test      pytest durations-*.json (python/mojolearn/tests/conftest.py; default ~/mojolearn-evidence/test-durations)
    group     tools/development_build.py group results <mode>-<stem>.json / support-<name>.json (`seconds`, state,
              peak_rss_bytes; default ~/mojolearn-evidence/development-builds)
    step      tools/release.py state.json steps (`at` = end, `seconds`; default ~/mojolearn-evidence/release/*/)
    lane      identity records and verifier reports through tools/identity_lane_timings.collect (CellTimer sums)
    job       box job persist.log START/EXIT lines (tools/box_persist_run.sh; fetched results dirs)
    workflow  saved `gh run view --json databaseId,workflowName,headSha,jobs` files (release_github_build.py's
              run.json), GitHub's startedAt/completedAt per job

A default directory that does not exist is skipped. A unit with only a duration gets start = end - seconds (its
file's mtime or recorded end), so each kind's share of wall time is close, not exact. Parallel units (xdist
tests, chunks) are summed (`sum_s`, worker-seconds) and also counted once (`wall_s`, the union of intervals).

Tests: tools/test_timing_ledger.py.
"""
import argparse
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = Path.home() / "mojolearn-evidence"
KINDS = ("test", "group", "step", "lane", "job", "workflow")
DEFAULTS = dict(tests=[EVIDENCE / "test-durations"], builds=[EVIDENCE / "development-builds"],
                release=sorted((EVIDENCE / "release").glob("*/state.json")) if (EVIDENCE / "release").is_dir() else [],
                identity=[], jobs=[], workflow=[])


def unit(kind, name, start, end, **extra):
    return dict(kind=kind, name=str(name), start=float(start), end=float(end), seconds=float(end) - float(start),
                extra=extra)


def _iso(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp() if text else None


def _json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------- readers (the files each tool already writes)
def read_tests(roots):
    out = []
    for root in roots:
        for path in sorted(Path(root).rglob("durations-*.json")) if Path(root).is_dir() else []:
            data = _json(path)
            if not isinstance(data, dict):
                continue
            end = path.stat().st_mtime
            for nodeid, e in (data.get("tests") or {}).items():
                out.append(unit("test", nodeid, end - float(e["seconds"]), end, outcome=e.get("outcome"),
                                file=nodeid.split("::", 1)[0], source=str(path)))
    return out


def read_builds(roots):
    out = []
    for root in roots:
        for path in sorted(Path(root).rglob("*.json")) if Path(root).is_dir() else []:
            if path.name.endswith(".execution.json"):
                continue
            data = _json(path)
            if not (isinstance(data, dict) and isinstance(data.get("seconds"), (int, float))
                    and ("stem" in data or "support" in data) and "key" in data):
                continue
            end = path.stat().st_mtime
            out.append(unit("group", path.stem, end - data["seconds"], end, state=data.get("state"),
                            rc=data.get("exit_code"), peak_rss_bytes=data.get("peak_rss_bytes"), source=str(path)))
    return out


def read_release(paths):
    out = []
    for raw in paths:
        p = Path(raw)
        state = p / "state.json" if p.is_dir() else p
        data = _json(state)
        if not isinstance(data, dict):
            continue
        for step, rec in sorted((data.get("steps") or {}).items()):
            if isinstance(rec, dict) and rec.get("seconds") is not None and rec.get("at"):
                end = _iso(rec["at"])
                out.append(unit("step", step, end - float(rec["seconds"]), end, release=str(state.parent),
                                result=str(rec.get("result", ""))[:200]))
    return out


def read_identity(paths):
    if not paths:
        return []
    sys.path.insert(0, str(ROOT / "tools"))
    import identity_lane_timings
    lanes, _read, _skipped = identity_lane_timings.collect([str(p) for p in paths])
    out = []
    for (column, lane), e in sorted(lanes.items()):
        if e["wall_s"] is None:
            continue
        end = max(Path(s).stat().st_mtime for s in e["sources"])
        out.append(unit("lane", lane, end - e["wall_s"], end, column=column, train_s=e["train_s"],
                        other_s=e["other_s"]))
    return out


_PLOG = re.compile(r"^(\S+Z) (START|EXIT) (.*)$")


def read_jobs(roots):
    """Box jobs from persist.log: `<utc> START tag=..` and `<utc> EXIT rc=..` (tools/box_persist_run.sh)."""
    out = []
    for root in roots:
        for path in sorted(Path(root).rglob("persist.log")) if Path(root).is_dir() else []:
            start = tag = None
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                m = _PLOG.match(line)
                if not m:
                    continue
                if m.group(2) == "START":
                    start = _iso(m.group(1))
                    t = re.search(r"tag=(\S+)", m.group(3))
                    tag = t.group(1) if t else path.parent.name
                elif start is not None:
                    rc = re.search(r"rc=(\d+)", m.group(3))
                    out.append(unit("job", tag, start, _iso(m.group(1)), rc=int(rc.group(1)) if rc else None,
                                    source=str(path)))
                    start = None
    return out


def read_workflows(paths):
    out = []
    for raw in paths:
        run = _json(raw)
        if not isinstance(run, dict):
            continue
        for job in run.get("jobs") or []:
            s, e = _iso(job.get("startedAt")), _iso(job.get("completedAt"))
            if s is None or e is None or e < s:
                continue
            out.append(unit("workflow", f"{run.get('workflowName') or 'workflow'}: {job.get('name', '?')}", s, e,
                            rc=0 if job.get("conclusion") in ("success", "skipped") else 1,
                            conclusion=job.get("conclusion"), run=run.get("databaseId") or run.get("url")))
    return out


READERS = dict(tests=read_tests, builds=read_builds, release=read_release, identity=read_identity,
               jobs=read_jobs, workflow=read_workflows)


def collect(sources):
    rows = []
    for name, reader in READERS.items():
        rows += reader(sources.get(name) or [])
    return rows


# ---------------------------------------------------------------- report
def parse_since(text, now=None):
    now = time.time() if now is None else now
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([smhd])", text.strip())
    if not m:
        raise ValueError(f"--since wants <number><s|m|h|d>, got {text!r}")
    return now - float(m.group(1)) * dict(s=1, m=60, h=3600, d=86400)[m.group(2)]


def union_seconds(intervals):
    """Wall time covered by the intervals (parallel units counted once)."""
    total, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def group_key(row, by):
    if by == "kind":
        return row["kind"]
    if by == "file":
        return row["extra"].get("file") or row["name"].split("::", 1)[0]
    return row["name"]


def failed(row):
    return row["extra"].get("rc") not in (None, 0) or row["extra"].get("outcome") in ("failed", "error")


def summarize(rows, kind=None, since=None, top=50, by="name"):
    rows = [r for r in rows if (kind is None or r["kind"] == kind) and (since is None or r["end"] >= since)]
    wall_all = union_seconds([(r["start"], r["end"]) for r in rows])
    kinds = {}
    for k in sorted({r["kind"] for r in rows}):
        mine = [r for r in rows if r["kind"] == k]
        wall = union_seconds([(r["start"], r["end"]) for r in mine])
        kinds[k] = dict(count=len(mine), sum_s=round(sum(r["seconds"] for r in mine), 1), wall_s=round(wall, 1),
                        wall_share=round(wall / wall_all, 3) if wall_all else None)
    groups = defaultdict(lambda: dict(count=0, sum_s=0.0, kinds=set(), failed=0))
    for r in rows:
        g = groups[group_key(r, by)]
        g["count"] += 1
        g["sum_s"] += r["seconds"]
        g["kinds"].add(r["kind"])
        g["failed"] += failed(r)
    ranked = sorted(groups.items(), key=lambda kv: -kv[1]["sum_s"])[:top]
    items = [dict(key=k, count=g["count"], sum_s=round(g["sum_s"], 1), kinds=sorted(g["kinds"]), failed=g["failed"])
             for k, g in ranked]
    return dict(records=len(rows), wall_s=round(wall_all, 1), sum_s=round(sum(r["seconds"] for r in rows), 1),
                kinds=kinds, top=items, by=by)


def print_report(rep, out=None):
    out = sys.stdout if out is None else out
    print(f"timing: {rep['records']} unit(s), {rep['sum_s']:.1f} s summed, "
          f"{rep['wall_s']:.1f} s wall (parallel units counted once)", file=out)
    print(f"{'kind':<10} {'count':>7} {'sum_s':>11} {'wall_s':>10} {'wall share':>10}", file=out)
    for k, v in rep["kinds"].items():
        share = "-" if v["wall_share"] is None else f"{100 * v['wall_share']:.1f}%"
        print(f"{k:<10} {v['count']:>7} {v['sum_s']:>11.1f} {v['wall_s']:>10.1f} {share:>10}", file=out)
    print(f"top {len(rep['top'])} by summed seconds (--by {rep['by']}):", file=out)
    for it in rep["top"]:
        fail = f"  FAILED {it['failed']}" if it["failed"] else ""
        print(f"{it['sum_s']:>10.1f} s  x{it['count']:<5} {','.join(it['kinds']):<8} {it['key'][:150]}{fail}",
              file=out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report")
    r.add_argument("--kind", choices=KINDS)
    r.add_argument("--since", help="e.g. 24h, 90m, 7d")
    r.add_argument("--top", type=int, default=50)
    r.add_argument("--by", choices=("name", "file", "kind"), default="name")
    r.add_argument("--json", action="store_true")
    for name in READERS:
        r.add_argument("--" + name, action="append", metavar="PATH",
                       help="replaces the default source(s) for this kind" if DEFAULTS[name] else None)
    r.add_argument("--no-defaults", action="store_true", help="read only the sources given on the command line")
    a = ap.parse_args(argv)
    sources = {name: (getattr(a, name) or ([] if a.no_defaults else DEFAULTS[name])) for name in READERS}
    rep = summarize(collect(sources), kind=a.kind, since=parse_since(a.since) if a.since else None,
                    top=a.top, by=a.by)
    if a.json:
        json.dump(rep, sys.stdout, indent=1)
        print()
    else:
        print_report(rep)
    return 0 if rep["records"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
