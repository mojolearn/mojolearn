#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Fold an M3 IDENTICAL sweep (~/mq/out/race-ident-*) into the canonical M3
board: replace each race's `ours` IDENTICAL cell, recompute ratios, re-render.

    python3 tools/af_board_ident_update.py BOARD_DIR SWEEP_GLOB HEAD [--dry-run]

Box (M3). Keeps BOARD_DIR/board.before-ident-<date>.json. Opponent and FAST
cells are untouched. A failed sweep retains its old measurement as history, but is not shown as current.
Includes extra_races and never replaces a newer or already imported sweep.
Then re-renders BOARD.md, BOARD_M3_FAST.md and BOARD_M3_IDENTICAL.md (tools/af_board_render.py)
and runs the render check.
"""
import copy, glob, json, os, shutil, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_board as bb

bdir, pat, head = sys.argv[1], sys.argv[2], sys.argv[3]
dry = "--dry-run" in sys.argv
path = os.path.join(bdir, "board.json")
board = json.load(open(path))
index = {}
records = dict(board["races"])
records.update(board.get("extra_races") or {})
for rid, rec in records.items():
    for c in rec.get("cells", []):
        if c.get("library") == "mojolearn" and c.get("mode") == "identical":
            index[(c["lane"], c["dataset"])] = (rid, c)
done, kept, missing = [], [], []
skipped = 0
for d in sorted(glob.glob(pat)):
    res = glob.glob(os.path.join(d, "A-1", "res", "*.json"))
    if not res:
        missing.append(os.path.basename(d)); continue
    r = json.load(open(res[0])); o = r["arms"].get("ours") or {}
    key = (r["lane"], r["dataset"])
    if key not in index:
        missing.append("%s/%s (no board race)" % key); continue
    rid, c = index[key]
    src = c.get("source") if isinstance(c.get("source"), dict) else {}
    if src.get("sweep") == os.path.basename(d) or (src.get("finished") and r.get("finished")
            and src["finished"] > r["finished"]):
        skipped += 1
        continue
    previous = copy.deepcopy(c)
    old = c.get("median_ms")
    if o.get("status") != "ok" or not o.get("median_ms"):
        c.update(status="stale: latest IDENTICAL sweep " + str(o.get("status", "missing time")),
                 median_ms=None, min_ms=None, max_ms=None, times_ms=[], rounds=0,
                 quality={}, hash=None,
                 source={"sweep": os.path.basename(d), "head": head, "finished": r.get("finished"),
                         "previous_measurement": previous, "failure": o})
        c.pop("quality_text", None)
        records[rid].setdefault("identical_page", {})["status"] = c["status"]
        bb.add_ratios(records[rid]["cells"])
        kept.append("%s/%s" % key)
        continue
    # status too: a cell that was REFUSED in 0.8.34 and now has a time is ok (it was left REFUSED before)
    c.update(status="ok", median_ms=o["median_ms"], min_ms=min(o["ms"]), max_ms=max(o["ms"]), rounds=len(o["ms"]),
             times_ms=o["ms"], quality=(r.get("quality") or {}).get("ours", c.get("quality")),
             hash=(o.get("digests") or [c.get("hash")])[-1],
             source={"sweep": os.path.basename(d), "head": head, "finished": r.get("finished"),
                     "previous_median_ms": old, "previous_hash": c.get("hash"),
                     "previous_status": previous.get("status"), "history": [src] if src else []})
    c.pop("quality_text", None)
    if records[rid].get("identical_page", {}).get("status", "").startswith("stale:"):
        records[rid]["identical_page"]["status"] = "ok"
    bb.add_ratios(records[rid]["cells"])
    done.append((key, old, o["median_ms"]))
print("updated=%d failed_stale=%d unmatched=%d skipped_current_or_newer=%d" % (len(done), len(kept), len(missing), skipped))
for k in kept: print("FAILED_STALE", k)
for m in missing: print("UNMATCHED", m)
if not dry:
    backup = os.path.join(bdir, "board.before-ident-%s.json" % time.strftime("%Y%m%d-%H%M%S"))
    shutil.copy2(path, backup)
    board["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    board.setdefault("box_history", []).append({"ident_sweep": pat, "head": head, "updated_cells": len(done)})
    with open(path, "w") as fh:
        json.dump(board, fh, indent=1)
    # board.json is the one M3 board: re-render BOARD.md and both docs pages, then check them
    import af_board_render as R
    R.write_all(bdir, R.DOCS, board)
    bad = R.check(bdir, R.DOCS)
    for b in bad:
        print(b)
    if bad:
        sys.exit("RENDER CHECK FAIL: the pages do not match board.json")
    print("RENDER CHECK OK")
