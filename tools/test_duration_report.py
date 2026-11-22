#!/usr/bin/env python3
"""Sum a test run's per-test durations: where the suite's time goes.

Reads the JSON files python/mojolearn/tests/conftest.py writes when
MOJOLEARN_TEST_DURATIONS_DIR is set (one per pytest process; the xdist
controller's holds every worker's tests), and prints:

* total test seconds (worker-seconds), split by outcome;
* the heaviest files (pytest-xdist `--dist loadfile` keeps a file on one
  worker, so a file's total is the unit the scheduler balances);
* the heaviest tests;
* an estimate of the wall time `--dist loadfile` needs on N workers for these
  file totals (longest-first assignment; xdist hands files out as workers
  free up, and each worker also pays its own startup, so the real run is
  somewhat longer).

    python3 tools/test_duration_report.py DIR [--workers 10] [--top 40] [--tsv OUT]
"""
import argparse
import heapq
import json
import sys
from collections import defaultdict
from pathlib import Path


def load(directory):
    tests = {}
    for path in sorted(Path(directory).glob("durations-*.json")):
        for nodeid, entry in json.loads(path.read_text(encoding="utf-8"))["tests"].items():
            tests[nodeid] = entry
    return tests


def file_totals(tests):
    totals = defaultdict(float)
    for nodeid, entry in tests.items():
        totals[nodeid.split("::", 1)[0]] += entry["seconds"]
    return dict(totals)


def loadfile_wall(totals, workers):
    """Makespan of longest-first assignment of file totals to `workers`."""
    loads = [0.0] * max(1, workers)
    heapq.heapify(loads)
    for seconds in sorted(totals.values(), reverse=True):
        heapq.heappush(loads, heapq.heappop(loads) + seconds)
    return max(loads)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--tsv", help="write every test as seconds<TAB>outcome<TAB>nodeid, heaviest first")
    args = ap.parse_args(argv)
    tests = load(args.directory)
    if not tests:
        print(f"no durations-*.json under {args.directory}", file=sys.stderr)
        return 2
    by_outcome = defaultdict(lambda: [0, 0.0])
    for entry in tests.values():
        by_outcome[entry["outcome"]][0] += 1
        by_outcome[entry["outcome"]][1] += entry["seconds"]
    total = sum(entry["seconds"] for entry in tests.values())
    print(f"tests {len(tests)}  test-seconds {total:.0f} ({total / 60:.1f} worker-min)")
    for outcome, (count, seconds) in sorted(by_outcome.items()):
        print(f"  {outcome:8s} {count:6d} tests {seconds:8.0f} s")
    totals = file_totals(tests)
    print(f"loadfile wall on {args.workers} workers (file totals, longest first): "
          f"{loadfile_wall(totals, args.workers) / 60:.1f} min")
    print(f"heaviest {args.top} files:")
    for name, seconds in sorted(totals.items(), key=lambda kv: -kv[1])[:args.top]:
        print(f"  {seconds:8.1f} s  {name}")
    ranked = sorted(tests.items(), key=lambda kv: -kv[1]["seconds"])
    print(f"heaviest {args.top} tests:")
    for nodeid, entry in ranked[:args.top]:
        print(f"  {entry['seconds']:8.2f} s  {entry['outcome']:7s} {nodeid[:200]}")
    if args.tsv:
        with open(args.tsv, "w", encoding="utf-8") as handle:
            for nodeid, entry in ranked:
                handle.write(f"{entry['seconds']:.3f}\t{entry['outcome']}\t{nodeid}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
