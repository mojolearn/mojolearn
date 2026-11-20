#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Per-lane wall time of identity records and verifier reports, longest first.

    python3 tools/identity_lane_timings.py <records dir | record.json>... [--top N] [--json]

Reads what the records already store; it runs nothing and changes nothing.

* An identity record (`tools/identity_break.py --json`, `<label>.identical.json`)
  times every cell (lane/fixture) with a monotonic clock (`CellTimer`,
  `time.perf_counter`): `cells[<lane>/<fixture>].timing.total_seconds` plus its
  `stages`. A lane's `wall_s` is the sum over its fixtures; `train_s` is the
  sum of its `*/train` stages and `other_s` the rest (infer, save-reload,
  batch and the other parts). A lane loads its binding inside its first fit,
  so binding load sits in `train_s`; the record cannot separate it.
* A verifier report (`mojolearn.verify-all-report.v1`) stores `lane_seconds`
  per lane; it has no stage split.

Timings are informational. No digest, closure, reuse key, comparison or
admission reads them, and a record without them (records older than the cell
timer) is reported as untimed, never as wrong. A directory argument is
searched for `*.json` files that are one of the two kinds; other JSON is
skipped. When several inputs hold the same cell of the same column (a resumed
record and its checkpoint, a part and its merge), the cell is counted once,
from the input given last.
"""
import argparse
import json
import sys
from pathlib import Path

VERIFY_FORMAT = "mojolearn.verify-all-report.v1"


def _files(paths):
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            yield from sorted(p for p in path.rglob("*.json") if p.is_file())
        else:
            yield path


def _column(data):
    if data.get("format") == VERIFY_FORMAT:
        device = data.get("device") or {}
        return str(device.get("vendor") or device.get("device_class") or "unknown")
    return str(data.get("vendor") or "unknown")


def collect(paths):
    """{(column, lane): entry} with wall_s/train_s/other_s, timed and untimed
    cell counts, and the files read. Entries with no timed cell have
    wall_s None."""
    cells = {}          # (column, lane, fixture) -> (timing or None, source)
    lane_reports = {}   # (column, lane) -> (seconds, source)   verifier reports
    read, skipped = [], []
    for path in _files(paths):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            skipped.append(f"{path}: {type(exc).__name__}")
            continue
        if not isinstance(data, dict):
            skipped.append(f"{path}: not an object")
            continue
        column = _column(data)
        if data.get("format") == VERIFY_FORMAT and isinstance(data.get("lane_seconds"), dict):
            for lane, seconds in data["lane_seconds"].items():
                lane_reports[(column, lane)] = (float(seconds), str(path))
            read.append(str(path))
        elif isinstance(data.get("cells"), dict) and "fixtures" in data:
            for key, cell in data["cells"].items():
                lane, _, fixture = key.partition("/")
                timing = cell.get("timing") if isinstance(cell, dict) else None
                if not (isinstance(timing, dict) and isinstance(timing.get("total_seconds"), (int, float))):
                    timing = None
                cells[(column, lane, fixture)] = (timing, str(path))
            read.append(str(path))
        else:
            skipped.append(f"{path}: neither an identity record nor a verifier report")

    lanes = {}

    def entry(column, lane):
        return lanes.setdefault((column, lane), dict(
            lane=lane, column=column, wall_s=None, train_s=None, other_s=None,
            timed_cells=0, untimed_cells=0, sources=set()))

    for (column, lane, _fixture), (timing, source) in cells.items():
        e = entry(column, lane)
        e["sources"].add(source)
        if timing is None:
            e["untimed_cells"] += 1
            continue
        total = float(timing["total_seconds"])
        train = sum(float(s) for name, s in (timing.get("stages") or {}).items()
                    if name.rsplit("/", 1)[-1] == "train" and isinstance(s, (int, float)))
        e["timed_cells"] += 1
        e["wall_s"] = (e["wall_s"] or 0.0) + total
        e["train_s"] = (e["train_s"] or 0.0) + train
        e["other_s"] = (e["other_s"] or 0.0) + max(total - train, 0.0)
    for (column, lane), (seconds, source) in lane_reports.items():
        # A verifier report and an identity record of the same column are
        # different runs; keep them apart rather than adding them.
        e = entry(f"{column} (verify)", lane)
        e["sources"].add(source)
        e["wall_s"] = seconds
    for e in lanes.values():
        e["sources"] = sorted(e["sources"])
    return lanes, read, skipped


def summarize(paths, top=20):
    lanes, read, skipped = collect(paths)
    timed = [e for e in lanes.values() if e["wall_s"] is not None]
    timed.sort(key=lambda e: (-e["wall_s"], e["column"], e["lane"]))
    untimed = sorted((e for e in lanes.values() if e["wall_s"] is None),
                     key=lambda e: (e["column"], e["lane"]))
    by_column = {}
    for e in timed:
        by_column[e["column"]] = by_column.get(e["column"], 0.0) + e["wall_s"]
    return dict(
        files_read=read, files_skipped=skipped,
        lanes=timed, untimed_lanes=[dict(lane=e["lane"], column=e["column"]) for e in untimed],
        total_wall_s=sum(e["wall_s"] for e in timed),
        column_wall_s=dict(sorted(by_column.items())),
        top=[dict(lane=e["lane"], column=e["column"], wall_s=e["wall_s"]) for e in timed[:top]],
    )


def _print_text(report, top):
    rows = report["lanes"]
    width = max([len(r["lane"]) for r in rows] + [4])
    cwidth = max([len(r["column"]) for r in rows] + [6])
    print(f"{'lane':<{width}}  {'column':<{cwidth}}  {'wall_s':>10}  {'train_s':>10}  {'other_s':>10}  cells")
    for r in rows:
        train = "" if r["train_s"] is None else f"{r['train_s']:.3f}"
        other = "" if r["other_s"] is None else f"{r['other_s']:.3f}"
        cells = f"{r['timed_cells']}" + (f"+{r['untimed_cells']} untimed" if r["untimed_cells"] else "")
        print(f"{r['lane']:<{width}}  {r['column']:<{cwidth}}  {r['wall_s']:>10.3f}  {train:>10}  {other:>10}  {cells}")
    print()
    for column, seconds in report["column_wall_s"].items():
        print(f"column {column}: {seconds:.3f} s")
    print(f"TOTAL {report['total_wall_s']:.3f} s over {len(rows)} timed lane(s); "
          f"{len(report['untimed_lanes'])} untimed; {len(report['files_read'])} file(s) read, "
          f"{len(report['files_skipped'])} skipped")
    print(f"TOP {min(top, len(rows))}:")
    for i, r in enumerate(report["top"], 1):
        print(f"{i:>3}. {r['lane']:<{width}}  {r['column']:<{cwidth}}  {r['wall_s']:>10.3f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="identity records, verifier reports, or directories of them")
    parser.add_argument("--top", type=int, default=20, help="how many of the longest lanes to list (default 20)")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args(argv)
    report = summarize(args.paths, top=args.top)
    if args.json:
        json.dump(report, sys.stdout, indent=1)
        print()
    else:
        _print_text(report, args.top)
    return 0 if report["files_read"] else 1


if __name__ == "__main__":
    sys.exit(main())
