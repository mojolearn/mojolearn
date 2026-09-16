#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Summarize identity stage times and scheduler waits without rerunning tests."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def summarize(paths):
    stages = defaultdict(float)
    wait = run = 0.0
    cells = {}
    for path in paths:
        data = json.loads(Path(path).read_text())
        if "wait_seconds" in data:
            wait += data["wait_seconds"]
            run += data["run_seconds"]
        for name, cell in data.get("cells", {}).items():
            # A resumed record can overlap its earlier checkpoint. Keep one.
            cells[(data.get("vendor"), data.get("commit"), name)] = cell
    for cell in cells.values():
        for stage, seconds in cell.get("timing", {}).get("stages", {}).items():
            stages[stage.split("/", 1)[-1]] += seconds
    return dict(wait_seconds=wait, run_seconds=run,
                timed_cells=sum("timing" in c for c in cells.values()),
                untimed_cells=sum("timing" not in c for c in cells.values()),
                stage_seconds=dict(sorted(stages.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", nargs="+")
    print(json.dumps(summarize(parser.parse_args().records), indent=2))
