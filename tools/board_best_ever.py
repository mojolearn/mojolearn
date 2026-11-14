#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn
"""tools/board_best_ever.py -- the best time our arm ever had on every board race, from every board ever committed.

Why (Andrew, 2026-11-13): the paper compare judges a row against 0.8.55 and 0.8.66 only. The history audit of
2026-11-13 mined 189 historical board blobs and found the biggest loss outside them (RF fit 5.0 s at 0.8.25 -> 30 s
since Oct 5); 49 rows had no measurement since 0.8.25-0.8.44. This tool productizes that miner
(~/mojolearn-evidence/briefs-20261113/history-audit/hist-trees-neighbors-cluster/scripts/mine_boards.py).

  mine [--repo DIR] [--out-dir DIR] [--jobs N] [--rev-args ...]
        walks every version of every board.json / board.json.gz and every paper compare / recipes JSON ever
        committed under bench/results/bench_board, bench/results/board-archive and bench/results/paper_compare
        (`git log --all --reflog -- <dirs>`: blobs later trimmed or deleted are included), collects our identical
        arm cells (vendor, device, race key = race id + phase + arm + batch, ms, output hash, explicit parameters,
        release, blob commit and date) and writes bench/results/board_best_ever/{nvidia,amd}.json: per race key the
        best ms with its date, release, commit, hash, params and device, the best per device, the latest cell, and
        the full de-duplicated timeline. Parallel: one worker process per (commit, path) blob, worker count
        os.cpu_count() (--jobs overrides); each worker runs one `git show` and parses one blob.
  show --vendor V [--top N]
        the rows whose latest cell is slowest against its best (LATEST>BEST), one line each.

FLAGS (per race key, in the file)
  PARAMS-DIFFER   the best cell's explicit parameters (a shared key with another value) or output hash differ from
                  the newest cell that recorded them: the "loss" may be a parameter change, not a regression (the audit's dbscan,
                  gbdt-symmetric and gbdt-lossguide rows)
  PARAMS-UNKNOWN  the best cell (or every later cell) recorded no parameters: a same-params claim cannot be checked
                  (boards before the 2026-10-14 read-back kept none); the hash is still compared
  LATEST>BEST     the latest cell is more than 1.10x the best and PARAMS-DIFFER does not hold

tools/paper_compare.py compare --best-ever reads these files (the "best ever" column). Tests:
tools/test_board_best_ever.py.
"""
import argparse
import gzip
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SCHEMA = "mojolearn-board-best-ever/1"
OUT_DIR = os.path.join(REPO, "bench", "results", "board_best_ever")
SEARCH_DIRS = ("bench/results/bench_board", "bench/results/board-archive", "bench/results/paper_compare")
VENDORS = ("nvidia", "amd")
SLOWER = 1.10
NO_TIME = re.compile(r"^(REFUSED|FAILED|UNKNOWN|ERROR|TIMEOUT|SKIPPED|CRASHED)", re.I)
BOARD_PATH = re.compile(r"(^|/)board\.json(\.gz)?$")
COMPARE_PATH = re.compile(r"(^|/)paper_compare/compare-(nvidia|amd)\.json$")
RECIPES_PATH = re.compile(r"(^|/)paper_compare/recipes-(nvidia|amd)\.json$")
RELEASE = re.compile(r"(\d+\.\d+\.\d+(?:[.a-z0-9]*)?)")
SOURCE_BUILD = re.compile(r"main@([0-9a-f]{7,40})")


# ---------------------------------------------------------------------------------------------------------------
# cell extraction (also used by tools/paper_compare.py compare --best-ever on the new board)
# ---------------------------------------------------------------------------------------------------------------

def number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and v > 0


def vendor_of(*texts):
    """nvidia, amd, or None (Apple and CPU cells are not columns here)."""
    for t in texts:
        s = str(t or "").lower()
        if not s:
            continue
        if re.search(r"nvidia|l40|rtx|4090|a100|h100|h200|geforce|cuda|nvc", s):
            return "nvidia"
        if re.search(r"\bamd\b|amd-|mi3\d\d|instinct|radeon|rocm|do-amd", s):
            return "amd"
        if re.search(r"apple|m2|m3|metal", s):
            return None
    return None


def race_key(race_id, phase, arm, batch):
    """The same key tools/paper_compare.py recipe_key makes: race id, phase, arm, batch."""
    return " ".join(x for x in (race_id, phase, arm, batch or "") if x)


def params_of(race, cell):
    """(params dict or None, source): the explicit parameters our arm read back (cell.params.params), else the lane
    configuration's params (the cell's settings, then the race's)."""
    p = (cell.get("params") or {}) if isinstance(cell.get("params"), dict) else {}
    if isinstance(p.get("params"), dict):
        return p["params"], "cell"
    for src, lc in (("settings", (cell.get("settings") or {}).get("lane_config")),
                    ("race", (race or {}).get("lane_config"))):
        if isinstance(lc, dict) and isinstance(lc.get("params"), dict):
            return lc["params"], src
    return None, None


def params_differ(a, b):
    """The shared keys whose values differ (different read-back sources add keys; only shared keys judge)."""
    if not isinstance(a, dict) or not isinstance(b, dict):
        return []
    return sorted(k for k in set(a) & set(b) if json.dumps(a[k], sort_keys=True) != json.dumps(b[k], sort_keys=True))


def release_of(version):
    """'0.8.42' / 'main@8899093fc' / '0.8.42 (main@8899093fc)' from a cell's version string."""
    s = str(version or "")
    rel, src = RELEASE.search(s), SOURCE_BUILD.search(s)
    if rel and src:
        return "%s (main@%s)" % (rel.group(1), src.group(1)[:10])
    if rel:
        return rel.group(1)
    return ("main@" + src.group(1)[:10]) if src else (s[:40] or None)


def is_ours_identical(race, cell):
    arm = str(cell.get("arm") or "")
    if not arm.startswith("ours"):
        return False
    mode = cell.get("mode") or ((race or {}).get("our_arms") or {}).get(arm)
    return mode == "identical"


def cell_records(board, blob_vendor_hint=""):
    """Every timed identical cell of our arms in one board document, as flat records (no commit/date)."""
    box_gpu = ((board.get("box") or {}).get("gpu") or {})
    races = board.get("races") or {}
    if isinstance(races, list):
        races = {r.get("id"): r for r in races if isinstance(r, dict)}
    when = board.get("updated") or board.get("created")
    box_version = ((board.get("box") or {}).get("mojolearn") or {}).get("version")
    box_host_id = (((board.get("box") or {}).get("host_probe") or {}).get("host_id"))
    out = []
    for rid, race in races.items():
        if not rid or not isinstance(race, dict):
            continue
        for phase, kind in (("fit", "cells"), ("infer", "infer_cells")):
            for c in race.get(kind) or []:
                if not isinstance(c, dict) or not is_ours_identical(race, c):
                    continue
                ms = c.get("median_ms")
                if not number(ms) or NO_TIME.match(str(c.get("status") or "")):
                    continue
                device = c.get("hardware_key") or box_gpu.get("name") or c.get("device_name")
                vendor = vendor_of(c.get("hardware_key"), box_gpu.get("vendor"), box_gpu.get("name"),
                                   c.get("device_name"), blob_vendor_hint)
                if vendor not in VENDORS:
                    continue
                params, psrc = params_of(race, c)
                # the box the time was taken on (bench_board race_host / box.host_probe, 2026-11-14): paper_compare
                # derives a best-ever time flag only against a cell of the same host id
                host_id = (race.get("host") or {}).get("host_id") or box_host_id
                version = c.get("version") or c.get("library_version") or box_version
                out.append({"key": race_key(rid, phase, c.get("arm"), c.get("batch")), "race_id": rid,
                            "family": race.get("family"), "lane": race.get("lane"), "dataset": race.get("dataset"),
                            "phase": phase, "arm": c.get("arm"), "batch": c.get("batch"), "vendor": vendor,
                            "device": device, "ms": float(ms), "hash": c.get("hash"), "params": params,
                            "params_source": psrc, "version": version, "release": release_of(version),
                            "board_updated": when, "host_id": host_id})
    return out


def compare_records(doc):
    """The new column of a committed paper compare JSON (no hash, no params: the compare did not keep them)."""
    vendor = doc.get("vendor")
    if vendor not in VENDORS:
        return []
    device = ",".join(doc.get("new_gpus") or []) or None
    label = doc.get("new_label") or ",".join(doc.get("new_versions") or [])
    out = []
    for r in doc.get("rows") or []:
        ms = r.get("new_ms")
        if not number(ms) or not str(r.get("arm") or "").startswith("ours"):
            continue
        out.append({"key": r.get("key") or race_key(r["race_id"], r["phase"], r["arm"], r.get("batch")),
                    "race_id": r.get("race_id"), "family": r.get("family"), "lane": r.get("lane"),
                    "dataset": r.get("dataset"), "phase": r.get("phase"), "arm": r.get("arm"),
                    "batch": r.get("batch"), "vendor": vendor, "device": device, "ms": float(ms), "hash": None,
                    "params": None, "params_source": None, "version": label, "release": release_of(label),
                    "board_updated": None})
    return out


def recipe_records(doc):
    """The paper's 0.8.55 cells of a committed recipes JSON."""
    vendor = doc.get("vendor")
    if vendor not in VENDORS:
        return []
    gpu = (doc.get("source") or {}).get("gpu")
    when = (doc.get("source") or {}).get("updated")
    out = []
    for r in doc.get("recipes") or []:
        old = r.get("v0855") or {}
        ms = old.get("median_ms")
        if r.get("mode") != "identical" or not number(ms) or NO_TIME.match(str(old.get("status") or "")):
            continue
        version = old.get("library_version") or doc.get("paper_version")
        out.append({"key": r.get("key"), "race_id": r.get("race_id"), "family": r.get("family"),
                    "lane": r.get("lane"), "dataset": r.get("dataset"), "phase": r.get("phase"), "arm": r.get("arm"),
                    "batch": r.get("batch"), "vendor": vendor, "device": old.get("device_name") or gpu,
                    "ms": float(ms), "hash": old.get("hash"), "params": r.get("params"),
                    "params_source": "cell" if r.get("params") is not None else None, "version": version,
                    "release": release_of(version), "board_updated": when})
    return out


def blob_records(path, raw):
    """Records of one blob (path decides its kind)."""
    if path.endswith(".gz"):
        raw = gzip.decompress(raw)
    doc = json.loads(raw)
    if not isinstance(doc, dict):
        return []
    if COMPARE_PATH.search(path):
        return compare_records(doc)
    if RECIPES_PATH.search(path):
        return recipe_records(doc)
    return cell_records(doc, path)


# ---------------------------------------------------------------------------------------------------------------
# mining (parallel over blobs)
# ---------------------------------------------------------------------------------------------------------------

def git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args], capture_output=True).stdout


def wanted(path):
    return bool(BOARD_PATH.search(path) or COMPARE_PATH.search(path) or RECIPES_PATH.search(path))


def list_paths(repo, rev_args):
    names = git(repo, "log", *rev_args, "--format=", "--name-only", "--", *SEARCH_DIRS).decode(errors="replace")
    return sorted({p.strip() for p in names.splitlines() if p.strip() and wanted(p.strip())})


def _path_versions(job):
    repo, rev_args, path = job
    out = git(repo, "log", *rev_args, "--format=%H %cs", "--", path).decode(errors="replace")
    return [(line.split()[0], line.split()[1], path) for line in out.splitlines() if len(line.split()) == 2]


def _mine_blob(job):
    """(commit, date, path, records, error) of one blob; a deleting commit has no blob and gives no records."""
    repo, sha, date, path = job
    raw = git(repo, "show", "%s:%s" % (sha, path))
    if not raw:
        return sha, date, path, [], None
    try:
        recs = blob_records(path, raw)
    except Exception as exc:  # noqa: BLE001 -- one bad blob is recorded, never the end of the mine
        return sha, date, path, [], "%s: %s" % (type(exc).__name__, str(exc)[:160])
    for r in recs:
        r.update(commit=sha[:12], commit_date=date, path=path)
        r["date"] = (str(r.pop("board_updated") or "")[:10]) or date
    return sha, date, path, recs, None


def mine(repo, jobs=None, rev_args=("--all", "--reflog")):
    """(records, stats). Every independent unit (a path's history, then a blob) runs in the process pool."""
    jobs = jobs or os.cpu_count() or 1
    paths = list_paths(repo, list(rev_args))
    with ProcessPoolExecutor(jobs) as ex:
        versions = [v for vs in ex.map(_path_versions, [(repo, list(rev_args), p) for p in paths]) for v in vs]
        blobs = sorted(set(versions))
        results = list(ex.map(_mine_blob, [(repo, s, d, p) for s, d, p in blobs], chunksize=4))
    records, errors, used = [], [], 0
    for sha, date, path, recs, err in results:
        if err:
            errors.append({"commit": sha[:12], "path": path, "error": err})
        if recs:
            used += 1
        records.extend(recs)
    return records, {"paths": len(paths), "blobs": len(blobs), "blobs_with_cells": used, "cells": len(records),
                     "errors": errors, "workers": jobs}


def _entry(r):
    return {k: r.get(k) for k in ("ms", "date", "commit_date", "release", "version", "commit", "path", "hash",
                                  "params", "params_source", "device", "host_id")}


SUSPECT_FRACTION = 0.05


def suspect_reason(best, timeline):
    """Why a best-ever time is implausible, or None (2026-11-13: 0.8.25 bests such as a 0.164 ms radius-neighbors
    fit). A best below 5% of the median of the row's other recorded times is SUSPECT: kept and shown, never
    dropped, never called a regression baseline. A record that merely lacks params is not SUSPECT (it stays a
    possible regression, BEST-EVER-REGRESSION-UNCHECKED)."""
    others = sorted(r["ms"] for r in timeline if r is not best and number(r.get("ms")))
    if others:
        n = len(others)
        median = others[n // 2] if n % 2 else (others[n // 2 - 1] + others[n // 2]) / 2
        if best["ms"] < SUSPECT_FRACTION * median:
            return "best %.4g ms is below 5%% of the median %.4g ms of the other times" % (best["ms"], median)
    return None


def summarize(records, vendor):
    """{race key: row} of one vendor: best, best per device, latest, flags, timeline."""
    by_key = {}
    for r in records:
        if r["vendor"] == vendor and r.get("key"):
            by_key.setdefault(r["key"], []).append(r)
    rows = {}
    for key, recs in sorted(by_key.items()):
        # one timeline entry per distinct measurement: the same cell is carried by many later commits of a board
        seen = {}
        for r in sorted(recs, key=lambda x: (x["date"], x["commit_date"], x["commit"])):
            ident = (round(r["ms"], 6), r.get("hash"), r.get("release"), r.get("device"))
            seen.setdefault(ident, r)
        timeline = sorted(seen.values(), key=lambda x: (x["date"], x["commit_date"]))
        best = min(timeline, key=lambda x: (x["ms"], x["date"]))
        latest = timeline[-1]
        per_device = {}
        for r in timeline:
            d = r.get("device") or "?"
            if d not in per_device or r["ms"] < per_device[d]["ms"]:
                per_device[d] = _entry(r)
        # the current params and hash: the newest cell that recorded them (a compare JSON's cell has neither)
        cur_p = next((r for r in reversed(timeline) if isinstance(r.get("params"), dict)), None)
        cur_h = next((r for r in reversed(timeline) if r.get("hash")), None)
        pdiff = params_differ(best.get("params"), cur_p.get("params")) if cur_p else []
        hdiff = bool(best.get("hash") and cur_h and best["hash"] != cur_h["hash"])
        flags = []
        if pdiff or hdiff:
            flags.append("PARAMS-DIFFER")
        elif not isinstance(best.get("params"), dict) or cur_p is None:
            flags.append("PARAMS-UNKNOWN")
        suspect = suspect_reason(best, timeline)
        if suspect:
            flags.append("SUSPECT")
        ratio = latest["ms"] / best["ms"]
        if ratio > SLOWER and "PARAMS-DIFFER" not in flags:
            flags.append("LATEST>BEST")
        first = recs[0]
        rows[key] = {"key": key, "race_id": first.get("race_id"), "family": first.get("family"),
                     "lane": first.get("lane"), "dataset": first.get("dataset"), "phase": first.get("phase"),
                     "arm": first.get("arm"), "batch": first.get("batch"),
                     "best": dict(_entry(best), suspect=suspect), "best_by_device": per_device, "latest": _entry(latest),
                     "latest_over_best": ratio, "params_differ": pdiff, "hash_differs": hdiff, "flags": flags,
                     "timeline": [_entry(r) for r in timeline]}
    return rows


def write(records, stats, out_dir, repo):
    os.makedirs(out_dir, exist_ok=True)
    head = git(repo, "rev-parse", "HEAD").decode().strip()
    paths = {}
    for v in VENDORS:
        rows = summarize(records, v)
        doc = {"schema": SCHEMA, "vendor": v, "mined_at_head": head, "search_dirs": list(SEARCH_DIRS),
               "stats": dict(stats, rows=len(rows)), "thresholds": {"latest_over_best_above": SLOWER},
               "flag_counts": {f: sum(f in r["flags"] for r in rows.values())
                               for f in ("PARAMS-DIFFER", "PARAMS-UNKNOWN", "LATEST>BEST")},
               "rows": rows}
        p = os.path.join(out_dir, "%s.json" % v)
        with open(p, "w") as fh:
            json.dump(doc, fh, indent=1, sort_keys=True)
            fh.write("\n")
        paths[v] = (p, len(rows), doc["flag_counts"])
    return paths


def load(path):
    with open(path) as fh:
        doc = json.load(fh)
    if doc.get("schema") != SCHEMA:
        raise SystemExit("board_best_ever: %s is not a %s document" % (path, SCHEMA))
    return doc


def cmd_mine(a):
    records, stats = mine(a.repo, a.jobs, tuple(a.rev_args.split()))
    paths = write(records, stats, a.out_dir, a.repo)
    print("BEST-EVER paths=%d blobs=%d blobs_with_cells=%d cells=%d errors=%d workers=%d" % (
        stats["paths"], stats["blobs"], stats["blobs_with_cells"], stats["cells"], len(stats["errors"]),
        stats["workers"]))
    for v, (p, n, fc) in paths.items():
        print("BEST-EVER %s rows=%d %s -> %s" % (v, n, " ".join("%s=%d" % kv for kv in sorted(fc.items())), p))
    return 0


def cmd_show(a):
    doc = load(a.file or os.path.join(OUT_DIR, "%s.json" % a.vendor))
    rows = sorted(doc["rows"].values(), key=lambda r: -r["latest_over_best"])
    for r in rows[:a.top]:
        b, l = r["best"], r["latest"]
        print("%.2fx %s best %.4g ms %s %s %s | latest %.4g ms %s %s %s" % (
            r["latest_over_best"], r["key"], b["ms"], b["release"], b["date"], b["device"], l["ms"], l["release"],
            l["date"], " ".join(r["flags"])))
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="board_best_ever", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mine", help="every committed board -> bench/results/board_best_ever/{nvidia,amd}.json")
    m.add_argument("--repo", default=REPO)
    m.add_argument("--out-dir", default=OUT_DIR)
    m.add_argument("--jobs", type=int, default=None, help="worker processes (default os.cpu_count())")
    m.add_argument("--rev-args", default="--all --reflog", help="git log revision arguments (default: every ref "
                                                                "and the reflog)")
    s = sub.add_parser("show", help="rows whose latest cell is slowest against its best")
    s.add_argument("--vendor", required=True, choices=VENDORS)
    s.add_argument("--file", default=None)
    s.add_argument("--top", type=int, default=30)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    return {"mine": cmd_mine, "show": cmd_show}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
