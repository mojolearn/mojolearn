#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""THE TIMING LEDGER: one JSON line per unit of work the project runs, and one report that ranks them
(Andrew, 2026-11-07: "do we measure everything in time? ... so we know where we need to make improvements").

    python3 tools/timing_ledger.py report [--kind test|group|step|lane|job|workflow|compile] [--since 24h]
                                          [--top 50] [--by name|file|kind] [--ledger PATH]... [--tests-dir DIR]
    python3 tools/timing_ledger.py ingest-tests DIR             durations-*.json of one pytest run -> kind test
    python3 tools/timing_ledger.py lanes RECORD.json... [--column LABEL]   identity record cells -> kind lane
    python3 tools/timing_ledger.py workflow RUN_ID [--repo O/R] [--json FILE]   gh run jobs -> kind workflow
    python3 tools/timing_ledger.py merge PATH...                append fetched ledgers (box results), deduped by id
    python3 tools/timing_ledger.py record KIND NAME START END [--rc N] [--extra k=v]...

Shell wrappers use tools/timing_ledger.sh (start|end|sample), which writes the same line format.

Where the ledger is: $MOJOLEARN_TIMING_LEDGER; else, inside a box job (tools/box_persist_run.sh sets
MOJOLEARN_BOX_PERSIST=1 and MOJOLEARN_RESULTS_DIR), <results dir>/timing-ledger.jsonl so it reaches R2 with the
job; else ~/mojolearn-evidence/timing/ledger.jsonl. Inside a pytest test (PYTEST_CURRENT_TEST) with no
MOJOLEARN_TIMING_LEDGER nothing is written, so tools' own unit tests never touch the real ledger. `merge` folds a fetched box ledger into the Mac's.

One line (schema mojolearn.timing.v1):
    {"schema", "id", "kind", "name", "start", "end", "seconds", "host", "cores", "busy", "commit", "rc", "extra"}
start/end are Unix epoch seconds. busy is {"mean", "max", "samples"} busy cores (sampled while the unit ran) or
null. Every record carries the git commit and the host name. Nothing here keeps a second clock: each producer
hands over the clock it already has (pytest's report durations, development_build's group seconds,
release.py's step clock, identity records' CellTimer, the box wrapper's start/end, GitHub's job timestamps).
A unit with only a duration gets start = end - seconds and extra.start_derived = true.

Tests: tools/test_timing_ledger.py.
"""
import argparse
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections import defaultdict
from pathlib import Path

SCHEMA = "mojolearn.timing.v1"
ROOT = Path(__file__).resolve().parent.parent
KINDS = ("test", "group", "step", "lane", "job", "workflow", "compile")
DEFAULT_LEDGER = Path.home() / "mojolearn-evidence" / "timing" / "ledger.jsonl"
DEFAULT_TESTS_DIR = Path.home() / "mojolearn-evidence" / "test-durations"
INGESTED_MARK = ".timing-ledgered"


def ledger_path(env=None):
    env = os.environ if env is None else env
    explicit = env.get("MOJOLEARN_TIMING_LEDGER", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    if env.get("PYTEST_CURRENT_TEST"):
        return None   # a tool's own unit test never writes the real ledger (set MOJOLEARN_TIMING_LEDGER to test it)
    if env.get("MOJOLEARN_BOX_PERSIST") == "1" and env.get("MOJOLEARN_RESULTS_DIR"):
        return Path(env["MOJOLEARN_RESULTS_DIR"]) / "timing-ledger.jsonl"
    return DEFAULT_LEDGER


def host_cores():
    """Cores this process may use (affinity / container quota where the OS says)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


_COMMIT = {}


def git_commit(cwd=None):
    """The checkout's HEAD (cached per directory); MOJOLEARN_TIMING_COMMIT or MOJOLEARN_COMMIT win (a box tree
    shipped without .git sets one); 'unknown' when neither exists."""
    for name in ("MOJOLEARN_TIMING_COMMIT", "MOJOLEARN_COMMIT"):
        if os.environ.get(name, "").strip():
            return os.environ[name].strip()
    key = str(cwd or ROOT)
    if key not in _COMMIT:
        try:
            out = subprocess.run(["git", "-C", key, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
            _COMMIT[key] = out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else "unknown"
        except (OSError, subprocess.SubprocessError):
            _COMMIT[key] = "unknown"
    return _COMMIT[key]


def _proc_stat():
    with open("/proc/stat") as fh:
        fields = [int(x) for x in fh.readline().split()[1:]]
    idle = fields[3] + (fields[4] if len(fields) > 4 else 0)   # idle + iowait
    return sum(fields), idle


def busy_sample(interval=1.0):
    """Busy cores right now: on Linux from /proc/stat over `interval` seconds, elsewhere the sum of ps %cpu.
    None when neither works."""
    try:
        total0, idle0 = _proc_stat()
        time.sleep(interval)
        total1, idle1 = _proc_stat()
        span = total1 - total0
        if span <= 0:
            return None
        return round(os.cpu_count() * (1.0 - (idle1 - idle0) / span), 1)
    except (OSError, ValueError, IndexError):
        pass
    try:
        out = subprocess.run(["ps", "-A", "-o", "%cpu="], capture_output=True, text=True, timeout=10).stdout
        return round(sum(float(x) for x in out.split() if x.replace(".", "", 1).isdigit()) / 100.0, 1)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def busy_summary(samples):
    samples = [s for s in samples if s is not None]
    if not samples:
        return None
    return dict(mean=round(sum(samples) / len(samples), 1), max=max(samples), samples=len(samples))


class BusySampler:
    """Samples busy cores every `every` seconds on a daemon thread (the first sample after `first` seconds),
    so a record says busy cores of total for the time the unit ran, not at its end."""

    def __init__(self, every=60.0, first=30.0):
        self.every, self.first, self.samples = every, first, []
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, name="timing-ledger-busy", daemon=True)
        self._thread.start()
        return self

    def _run(self):
        wait = self.first
        while not self._stop.wait(wait):
            self.samples.append(busy_sample())
            wait = self.every

    def __exit__(self, *exc):
        self._stop.set()
        return False

    def summary(self):
        return busy_summary(self.samples)


def _line(kind, name, start, end, host=None, cores=None, busy=None, extra=None, rc=None, commit=None):
    if isinstance(busy, (int, float)):
        busy = dict(mean=float(busy), max=float(busy), samples=1)
    start = None if start is None else float(start)
    end = None if end is None else float(end)
    return dict(schema=SCHEMA, id=uuid.uuid4().hex, kind=str(kind), name=str(name), start=start, end=end,
                seconds=None if start is None or end is None else round(end - start, 3),
                host=host or socket.gethostname(), cores=host_cores() if cores is None else int(cores),
                busy=busy, commit=commit or git_commit(), rc=rc, extra=dict(extra or {}))


def _append(rows, ledger=None):
    path = Path(ledger) if ledger else ledger_path()
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    if not data:
        return path
    # one O_APPEND write per call: lines from parallel producers (release steps, xdist, chunks) never interleave
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, data.encode())
    finally:
        os.close(fd)
    return path


def record(kind, name, start, end, host=None, cores=None, busy=None, extra=None, rc=None, ledger=None, commit=None):
    """Append one unit of work. Returns the line written."""
    row = _line(kind, name, start, end, host=host, cores=cores, busy=busy, extra=extra, rc=rc, commit=commit)
    _append([row], ledger)
    return row


def record_seconds(kind, name, seconds, end=None, **kw):
    """A unit whose producer kept only a duration: start = end - seconds, marked start_derived."""
    end = time.time() if end is None else float(end)
    extra = dict(kw.pop("extra", None) or {}, start_derived=True)
    return record(kind, name, end - float(seconds), end, extra=extra, **kw)


def safe(fn, *args, **kw):
    """Producers call the ledger through this: timing is never a reason for a run to fail."""
    try:
        return fn(*args, **kw)
    except Exception as exc:   # noqa: BLE001 - the ledger is informational
        print(f"timing_ledger: not recorded ({type(exc).__name__}: {exc})", file=sys.stderr)
        return None


# ---------------------------------------------------------------- feeds (existing clocks -> ledger)
def ingest_tests(directory, ledger=None, force=False):
    """One record per test from a run's durations-*.json (python/mojolearn/tests/conftest.py). Each file's
    mtime is its session end; a test's start is derived from its seconds. A file is ingested once: the
    directory's INGESTED_MARK lists each ingested file as name:mtime_ns (conftest's default dir `latest` is
    rewritten by every bare pytest run, so a rewritten file is new). force re-ingests. Returns tests recorded."""
    directory = Path(directory)
    mark = directory / INGESTED_MARK
    done = set()
    if mark.exists() and not force:
        done = set(mark.read_text(encoding="utf-8").split())
    rows, keys = [], []
    for path in sorted(directory.glob("durations-*.json")):
        stat = path.stat()
        key = f"{path.name}:{stat.st_mtime_ns}"
        if key in done:
            continue
        end = stat.st_mtime
        tests = json.loads(path.read_text(encoding="utf-8")).get("tests", {})
        for nodeid, entry in tests.items():
            rows.append(_line("test", nodeid, end - float(entry["seconds"]), end,
                              extra=dict(outcome=entry.get("outcome"), file=nodeid.split("::", 1)[0],
                                         source=str(path), start_derived=True)))
        keys.append(key)
    if keys and (not rows or _append(rows, ledger) is not None):
        mark.write_text("\n".join(sorted(done | set(keys))) + "\n", encoding="utf-8")
    return len(rows)


def ingest_lanes(paths, column=None, ledger=None, end=None):
    """One record per (column, lane) from identity records / verifier reports, through
    tools/identity_lane_timings.collect (the CellTimer sums). Returns the number of lanes recorded."""
    sys.path.insert(0, str(ROOT / "tools"))
    import identity_lane_timings
    lanes, _read, _skipped = identity_lane_timings.collect([str(p) for p in paths])
    rows = []
    for (col, lane), e in sorted(lanes.items()):
        if e["wall_s"] is None:
            continue
        sources = e["sources"]
        stop = end if end is not None else max(Path(s).stat().st_mtime for s in sources)
        rows.append(_line("lane", lane, stop - e["wall_s"], stop,
                          extra=dict(column=column or col, train_s=e["train_s"], other_s=e["other_s"],
                                     timed_cells=e["timed_cells"], untimed_cells=e["untimed_cells"],
                                     sources=sources, start_derived=True)))
    _append(rows, ledger)
    return len(rows)


def _iso(text):
    from datetime import datetime
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp() if text else None


def ingest_workflow(run, ledger=None):
    """One record per job of a GitHub Actions run (`gh run view <id> --json ...` output as a dict)."""
    rows = []
    for job in run.get("jobs", []):
        start, end = _iso(job.get("startedAt")), _iso(job.get("completedAt"))
        if start is None or end is None or end < start:
            continue
        rows.append(_line("workflow", f"{run.get('workflowName', '?')}: {job.get('name', '?')}", start, end,
                          host="github", cores=0, commit=run.get("headSha") or "unknown",
                          rc=0 if job.get("conclusion") == "success" else 1,
                          extra=dict(run_id=run.get("databaseId"), conclusion=job.get("conclusion"),
                                     url=job.get("url"))))
    _append(rows, ledger)
    return len(rows)


def fetch_workflow(run_id, repo=None):
    argv = ["gh", "run", "view", str(run_id), "--json", "databaseId,workflowName,headSha,jobs"]
    if repo:
        argv += ["--repo", repo]
    return json.loads(subprocess.run(argv, capture_output=True, text=True, check=True).stdout)


def merge(paths, ledger=None):
    """Append other ledgers' lines (a fetched box results dir, a file) whose id this ledger lacks."""
    target = Path(ledger) if ledger else ledger_path()
    if target is None:
        return 0
    have = {r["id"] for r in load([target])} if target.exists() else set()
    new = [r for r in load(paths) if r["id"] not in have]
    _append(new, target)
    return len(new)


# ---------------------------------------------------------------- report
def _ledger_files(paths):
    for raw in paths:
        p = Path(raw).expanduser()
        if p.is_dir():
            yield from sorted(x for x in p.rglob("*.jsonl") if "ledger" in x.name)
        elif p.is_file():
            yield p


def load(paths):
    rows = []
    for path in _ledger_files(paths):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and row.get("schema") == SCHEMA and row.get("seconds") is not None:
                rows.append(row)
    seen, out = set(), []
    for r in rows:
        if r["id"] not in seen:
            seen.add(r["id"])
            out.append(r)
    return out


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
        return (row.get("extra") or {}).get("file") or row["name"].split("::", 1)[0].split(":", 1)[0]
    return row["name"]


def summarize(rows, kind=None, since=None, top=50, by="name"):
    rows = [r for r in rows if (kind is None or r["kind"] == kind)
            and (since is None or (r.get("end") or 0) >= since)]
    spans = [(r["start"], r["end"]) for r in rows if r.get("start") is not None and r.get("end") is not None]
    wall_all = union_seconds(spans)
    kinds = {}
    for k in sorted({r["kind"] for r in rows}):
        mine = [r for r in rows if r["kind"] == k]
        wall = union_seconds([(r["start"], r["end"]) for r in mine])
        kinds[k] = dict(count=len(mine), sum_s=round(sum(r["seconds"] for r in mine), 1), wall_s=round(wall, 1),
                        wall_share=round(wall / wall_all, 3) if wall_all else None)
    groups = defaultdict(lambda: dict(count=0, sum_s=0.0, kinds=set(), hosts=set(), busy_max=None, cores=None,
                                      failed=0))
    for r in rows:
        g = groups[group_key(r, by)]
        g["count"] += 1
        g["sum_s"] += r["seconds"]
        g["kinds"].add(r["kind"])
        g["hosts"].add(r.get("host") or "?")
        if r.get("rc") not in (None, 0) or (r.get("extra") or {}).get("outcome") in ("failed", "error"):
            g["failed"] += 1
        busy = r.get("busy")
        if isinstance(busy, dict) and busy.get("max") is not None:
            if g["busy_max"] is None or busy["max"] > g["busy_max"]:
                g["busy_max"], g["cores"] = busy["max"], r.get("cores")
    ranked = sorted(groups.items(), key=lambda kv: -kv[1]["sum_s"])
    items = [dict(key=key, count=g["count"], sum_s=round(g["sum_s"], 1), kinds=sorted(g["kinds"]),
                  hosts=sorted(g["hosts"]), failed=g["failed"], busy_max=g["busy_max"], cores=g["cores"])
             for key, g in ranked[:top]]
    return dict(records=len(rows), wall_s=round(wall_all, 1), sum_s=round(sum(r["seconds"] for r in rows), 1),
                kinds=kinds, top=items, by=by)


def print_report(rep, out=None):
    out = sys.stdout if out is None else out
    print(f"timing ledger: {rep['records']} record(s), {rep['sum_s']:.1f} s summed, "
          f"{rep['wall_s']:.1f} s wall (parallel units counted once)", file=out)
    print(f"{'kind':<10} {'count':>7} {'sum_s':>11} {'wall_s':>10} {'wall share':>10}", file=out)
    for k, v in rep["kinds"].items():
        share = "-" if v["wall_share"] is None else f"{100 * v['wall_share']:.1f}%"
        print(f"{k:<10} {v['count']:>7} {v['sum_s']:>11.1f} {v['wall_s']:>10.1f} {share:>10}", file=out)
    print(f"top {len(rep['top'])} by summed seconds (--by {rep['by']}):", file=out)
    for it in rep["top"]:
        busy = "" if it["busy_max"] is None else f"  busy {it['busy_max']} of {it['cores']}"
        fail = f"  FAILED {it['failed']}" if it["failed"] else ""
        print(f"{it['sum_s']:>10.1f} s  x{it['count']:<5} {','.join(it['kinds']):<8} {it['key'][:150]}"
              f"{busy}{fail}", file=out)


def _extra_pairs(pairs):
    out = {}
    for p in pairs or []:
        k, _, v = p.partition("=")
        out[k] = v
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report")
    r.add_argument("--kind", choices=KINDS)
    r.add_argument("--since", help="e.g. 24h, 90m, 7d")
    r.add_argument("--top", type=int, default=50)
    r.add_argument("--by", choices=("name", "file", "kind"), default="name")
    r.add_argument("--ledger", action="append", help="ledger file or directory (repeatable; default: the ledger)")
    r.add_argument("--tests-dir", default=str(DEFAULT_TESTS_DIR),
                   help="test-duration run dirs ingested first if not yet ledgered ('' to skip)")
    r.add_argument("--json", action="store_true")
    t = sub.add_parser("ingest-tests")
    t.add_argument("directory")
    t.add_argument("--force", action="store_true")
    la = sub.add_parser("lanes")
    la.add_argument("records", nargs="+")
    la.add_argument("--column")
    w = sub.add_parser("workflow")
    w.add_argument("run_id")
    w.add_argument("--repo")
    w.add_argument("--json", dest="json_file", help="a saved `gh run view --json databaseId,workflowName,headSha,jobs`")
    m = sub.add_parser("merge")
    m.add_argument("paths", nargs="+")
    rc = sub.add_parser("record")
    rc.add_argument("kind", choices=KINDS)
    rc.add_argument("name")
    rc.add_argument("start", type=float)
    rc.add_argument("end", type=float)
    rc.add_argument("--rc", type=int)
    rc.add_argument("--extra", action="append")
    a = ap.parse_args(argv)
    if a.cmd == "report":
        if a.tests_dir and Path(a.tests_dir).is_dir():
            for d in sorted(p for p in Path(a.tests_dir).iterdir() if p.is_dir()):
                safe(ingest_tests, d)
        paths = a.ledger or [p for p in [ledger_path()] if p]
        rep = summarize(load(paths), kind=a.kind, since=parse_since(a.since) if a.since else None,
                        top=a.top, by=a.by)
        if a.json:
            json.dump(rep, sys.stdout, indent=1)
            print()
        else:
            print_report(rep)
        return 0 if rep["records"] else 1
    if a.cmd == "ingest-tests":
        print(f"timing_ledger: {ingest_tests(a.directory, force=a.force)} test(s) -> {ledger_path()}")
        return 0
    if a.cmd == "lanes":
        print(f"timing_ledger: {ingest_lanes(a.records, column=a.column)} lane(s) -> {ledger_path()}")
        return 0
    if a.cmd == "workflow":
        try:
            run = json.loads(Path(a.json_file).read_text()) if a.json_file else fetch_workflow(a.run_id, a.repo)
            print(f"timing_ledger: {ingest_workflow(run)} workflow job(s) -> {ledger_path()}")
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            print(f"timing_ledger: run {a.run_id} not recorded ({type(exc).__name__}: {exc})")
        return 0
    if a.cmd == "merge":
        print(f"timing_ledger: {merge(a.paths)} record(s) merged -> {ledger_path()}")
        return 0
    if a.cmd == "record":
        record(a.kind, a.name, a.start, a.end, rc=a.rc, extra=_extra_pairs(a.extra))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
