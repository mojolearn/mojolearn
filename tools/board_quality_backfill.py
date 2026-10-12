#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Fill missing board quality from the board run's own raw race JSONs. Nothing is re-run.

A board cell can be stored with no quality although the race that produced it recorded one: the raw race
JSON (the racer's output, `<board>/raw/<family>/<rows>/<lane>-<dataset>.json`, named by the race record's
`race_json`) keeps a per-arm `quality` (fit) and `infer.quality` (inference) that ingest did not carry over.
This copies those values into cells that have no usable quality, and records where each came from in
`quality_source`, the convention of bench_board.stored_identity_quality.

Only the SAME run is used: --raw is the directory the board run wrote (the archive of that run), and a
cell is filled only from the race JSON its race record names. A cell that already has a numeric quality is
never touched. Usable = a finite number under a metric name that is not bookkeeping (output_shape, finite,
steps, ...), as on the published measurements page.

Usage:
  python3 tools/board_quality_backfill.py --board bench/results/bench_board/m3ultra-0834/board.json \
      --raw <unpacked run>/board/raw --source "R2 measurements/2026-10-02/board-0834-m3ultra.tar.gz" [--write]
Without --write it reports what it would fill.
"""
import argparse
import json
import math
import os
import sys

BOOKKEEPING = {"finite", "output_shape", "steps", "n_iter", "rows_with_repeated_ids", "n_clusters", "n_selected",
               "error"}


def usable(q):
    return {k: v for k, v in (q or {}).items()
            if k not in BOOKKEEPING and not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v)}


def per_arm(block):
    """A raw quality block keyed by arm ({"ours": {...}, "sklearn-cpu": {...}}), or {} when it is not."""
    if not isinstance(block, dict) or not block:
        return {}
    return {a: q for a, q in block.items() if isinstance(q, dict)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board", required=True)
    ap.add_argument("--raw", required=True, help="the raw/ directory of the run that built this board")
    ap.add_argument("--source", required=True, help="where --raw came from, recorded in quality_source")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    board = json.load(open(a.board))
    filled, missing_raw, no_value = [], 0, 0
    races = list(board.get("races", {}).items()) + list((board.get("extra_races") or {}).items())
    for rid, race in races:
        rel = race.get("race_json")
        if not rel:
            continue
        path = os.path.join(a.raw, rel[len("raw/"):] if rel.startswith("raw/") else rel)
        if not os.path.exists(path):
            missing_raw += 1
            continue
        raw = json.load(open(path))
        sources = {"fit": per_arm(raw.get("quality")), "infer": per_arm((raw.get("infer") or {}).get("quality"))}
        for phase, cells in (("fit", race.get("cells") or []), ("infer", race.get("infer_cells") or [])):
            for c in cells:
                if c.get("status") != "ok" or usable(c.get("quality")):
                    continue
                found = usable(sources[phase].get(c.get("arm")))
                if not found:
                    no_value += 1
                    continue
                old = c.get("quality")
                if a.write:
                    if old:
                        c["quality_superseded"] = old
                    c["quality"] = dict(old or {}, **found)
                    c["quality_source"] = "%s quality of arm %s in %s (%s): the run that built this cell" % (
                        "inference" if phase == "infer" else "fit", c.get("arm"), rel, a.source)
                filled.append((rid, phase, c.get("arm"), sorted(found)))
    for f in filled[:40]:
        print("fill", *f)
    print("board_quality_backfill: %d cells filled%s; %d still without a value in the raw JSON; %d races without "
          "a raw JSON" % (len(filled), "" if a.write else " (dry run)", no_value, missing_raw))
    if a.write and filled:
        tmp = a.board + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(board, fh, indent=1, sort_keys=False)
            fh.write("\n")
        os.replace(tmp, a.board)


if __name__ == "__main__":
    sys.exit(main())
